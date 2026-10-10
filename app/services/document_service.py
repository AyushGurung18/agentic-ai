"""
app/services/document_service.py
──────────────────────────────────
All DB writes related to documents, chunks, embeddings, and metadata.

This keeps rag.py clean — it just calls ingest_document() and gets back
the document_id without caring about the persistence details.

Flow
────
    ingest_document(text, filename, user_id, metadata_tags?)
        │
        ├─► INSERT INTO documents            → document_id
        │
        ├─► for each chunk:
        │     INSERT INTO document_chunks    → chunk_id
        │     INSERT INTO embeddings         (vector computed here)
        │
        └─► INSERT INTO document_metadata   (optional key/value tags)

The chunk texts + metadata ({"chunk_id": ..., "document_id": ...}) are
returned so rag.py can also hand them to PGVector for similarity search.
"""

import uuid
from typing import Optional

from app.db.database import get_conn
from app.services.embeddings import embed_text


def ingest_document(
    text: str,
    hierarchical_chunks: list[dict],
    filename: str,
    user_id: str,
    metadata_tags: Optional[dict[str, str]] = None,
    progress_callback = None,
    doc_type: str = "pdf",
    pages: Optional[list] = None,       # list of PageOffset (see pdf_loader/docx_loader)
    r2_url: Optional[str] = None,
    home_session_id: Optional[str] = None,
) -> tuple[str, list[dict]]:
    """
    Persist a document and its hierarchical chunks + embeddings to the DB.

    Parameters
    ----------
    text                : full raw text extracted from the file
    hierarchical_chunks : list of dicts {"parent": str, "children": [str, ...]}
    filename            : original file name
    user_id      : UUID string of the owning user
    metadata_tags: optional dict of key→value tags (e.g. {"section": "education"})

    Returns
    -------
    (document_id, chunk_metadatas)
        document_id    — UUID string of the newly created document row
        chunk_metadatas — list of dicts, one per chunk, to pass as PGVector metadata:
                          [{"chunk_id": "...", "document_id": "...", "chunk_index": 0}, ...]
    """
    # ── 1. Flatten children for embedding ────────────────────────────────────
    flat_children = []
    for p in hierarchical_chunks:
        flat_children.extend(p["children"])
        
    total_chunks = len(flat_children)
    vectors = []
    
    # Batch embed to allow progress reporting
    batch_size = 32
    for i in range(0, total_chunks, batch_size):
        batch = flat_children[i:i + batch_size]
        vectors.extend(embed_text(batch))
        
        if progress_callback:
            # 0% to 50% for embedding
            progress_callback(int((len(vectors) / total_chunks) * 50))
            
    vector_idx = 0

    with get_conn() as conn:
        with conn.cursor() as cur:

            # ── 2. Insert document ───────────────────────────────────────────
            cur.execute(
                """
                INSERT INTO documents (user_id, filename, content, doc_type, status, page_count, char_count, r2_url, home_session_id)
                VALUES (%s::uuid, %s, %s, %s, 'ready', %s, %s, %s, %s::uuid)
                RETURNING id
                """,
                # page_count is a PDF concept — DOCX always gets a single
                # synthetic PageOffset (see docx_loader.py) to keep the
                # document_pages table populated for both formats, but that's
                # not a real "page" and must not surface as page_count=1.
                (user_id, filename, text, doc_type, len(pages) if (pages and doc_type == "pdf") else None, len(text), r2_url, home_session_id),
            )
            document_id: str = str(cur.fetchone()[0])

            # ── 2b. Insert page offset map (citation highlighting) ───────────
            # `pages` crosses a Celery/JSON boundary by the time it gets here,
            # so it's always a list of plain dicts, never PageOffset instances.
            # Batched (one round trip for every page) rather than a per-page
            # execute() loop — matters for the 150-page documents this app is
            # explicitly meant to handle.
            if pages:
                cur.executemany(
                    """
                    INSERT INTO document_pages (document_id, page_number, char_start, char_end)
                    VALUES (%s::uuid, %s, %s, %s)
                    """,
                    [(document_id, p["page_number"], p["char_start"], p["char_end"]) for p in pages],
                )

            # ── 3. Insert chunks + embeddings ────────────────────────────────
            # IDs are generated client-side (instead of relying on the table's
            # gen_random_uuid() default + RETURNING) so every parent, child,
            # and embedding row can go through a single executemany() each —
            # 3 round trips total instead of up to 3 serial round trips *per
            # child chunk* (each previously blocked on a RETURNING + fetchone()
            # before the next INSERT could even be sent). For a 150-page
            # document with hundreds of chunks that serial pattern was the
            # actual cause of ingestion jobs blowing past the Celery
            # soft_time_limit — this keeps the chunk count the pipeline can
            # handle well above what any real contract will need.
            chunk_metadatas: list[dict] = []
            parent_rows: list[tuple] = []
            child_rows: list[tuple] = []
            embedding_rows: list[tuple] = []
            global_chunk_idx = 0

            for p in hierarchical_chunks:
                parent_id = str(uuid.uuid4())
                parent_rows.append((parent_id, document_id, global_chunk_idx, p["parent"]))
                global_chunk_idx += 1

                for child_text in p["children"]:
                    child_id = str(uuid.uuid4())
                    child_rows.append((child_id, document_id, parent_id, global_chunk_idx, child_text))
                    global_chunk_idx += 1

                    vector = vectors[vector_idx]
                    vector_idx += 1
                    embedding_rows.append((child_id, str(vector)))

                    chunk_metadatas.append({
                        "chunk_id": child_id,
                        "document_id": document_id,
                        "parent_chunk_id": parent_id,
                        "chunk_index": global_chunk_idx - 1,
                        "filename": filename,
                        "user_id": user_id,
                    })

            cur.executemany(
                """
                INSERT INTO document_chunks (id, document_id, chunk_index, content)
                VALUES (%s::uuid, %s::uuid, %s, %s)
                """,
                parent_rows,
            )
            if progress_callback:
                progress_callback(65)

            cur.executemany(
                """
                INSERT INTO document_chunks (id, document_id, parent_chunk_id, chunk_index, content)
                VALUES (%s::uuid, %s::uuid, %s::uuid, %s, %s)
                """,
                child_rows,
            )
            if progress_callback:
                progress_callback(85)

            cur.executemany(
                """
                INSERT INTO embeddings (chunk_id, embedding)
                VALUES (%s::uuid, %s::vector)
                """,
                embedding_rows,
            )
            if progress_callback:
                progress_callback(100)

            # ── 4. Insert metadata tags (optional) ──────────────────────────
            if metadata_tags:
                for key, value in metadata_tags.items():
                    cur.execute(
                        """
                        INSERT INTO document_metadata (document_id, key, value)
                        VALUES (%s::uuid, %s, %s)
                        """,
                        (document_id, key, value),
                    )

        conn.commit()

    print(
        f"✅ Ingested '{filename}': document_id={document_id}, "
        f"{len(hierarchical_chunks)} parents, {len(flat_children)} children"
    )
    return document_id, chunk_metadatas


# ── Session helpers ───────────────────────────────────────────────────────────

def ensure_session(session_id: str, user_id: str, title: Optional[str] = None) -> str:
    """
    Upsert a session row.  If it already exists, do nothing.
    Returns the session_id (same value passed in).
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (id, user_id, title)
                VALUES (%s::uuid, %s::uuid, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (session_id, user_id, title),
            )
        conn.commit()
    return session_id


def ensure_user(user_id: str, email: str = "dev@local", name: str = "Dev User") -> str:
    """
    Upsert a user row by id.  Used to bootstrap the default dev user.
    Returns the user_id.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (id, email, name)
                VALUES (%s::uuid, %s, %s)
                ON CONFLICT (id) DO NOTHING
                """,
                (user_id, email, name),
            )
        conn.commit()
    return user_id


# ── Document library ─────────────────────────────────────────────────────────

def list_documents(user_id: str) -> list[dict]:
    """All documents owned by a user, newest first — the document library."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, doc_type, status, page_count, char_count, created_at, home_session_id
                FROM   documents
                WHERE  user_id = %s::uuid
                ORDER  BY created_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
    return [
        {
            "id": str(r[0]), "filename": r[1], "doc_type": r[2], "status": r[3],
            "page_count": r[4], "char_count": r[5], "created_at": r[6],
            "home_session_id": str(r[7]) if r[7] else None,
        }
        for r in rows
    ]


def get_document(document_id: str, user_id: str) -> Optional[dict]:
    """Full document row (including raw content) — ownership-checked."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, filename, content, doc_type, status, page_count, char_count, r2_url, created_at
                FROM   documents
                WHERE  id = %s::uuid
                """,
                (document_id,),
            )
            row = cur.fetchone()
            if not row or str(row[1]) != user_id:
                return None

            cur.execute(
                """
                SELECT page_number, char_start, char_end
                FROM   document_pages
                WHERE  document_id = %s::uuid
                ORDER  BY page_number
                """,
                (document_id,),
            )
            pages = [{"page_number": p[0], "char_start": p[1], "char_end": p[2]} for p in cur.fetchall()]

    return {
        "id": str(row[0]), "filename": row[2], "content": row[3], "doc_type": row[4],
        "status": row[5], "page_count": row[6], "char_count": row[7], "r2_url": row[8],
        "created_at": row[9], "pages": pages,
    }


def get_documents_by_ids(document_ids: list[str], user_id: str) -> dict[str, dict]:
    """Batch fetch (content + page map) for quote verification / multi-doc chat.

    Ownership-checked — a document_id that doesn't belong to user_id is
    silently dropped rather than raising, since callers filter their own
    requested ID list against this result.
    """
    if not document_ids:
        return {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, filename, content, doc_type, status, created_at
                FROM   documents
                WHERE  id = ANY(%s::uuid[]) AND user_id = %s::uuid
                """,
                (document_ids, user_id),
            )
            rows = cur.fetchall()
            if not rows:
                return {}

            out: dict[str, dict] = {
                str(r[0]): {"filename": r[1], "content": r[2], "doc_type": r[3], "status": r[4], "created_at": r[5], "pages": []}
                for r in rows
            }

            # One query for every document's page map instead of one query
            # per document — matters once a multi-document question selects
            # several library documents at once.
            cur.execute(
                """
                SELECT document_id, page_number, char_start, char_end
                FROM   document_pages
                WHERE  document_id = ANY(%s::uuid[])
                ORDER  BY document_id, page_number
                """,
                (list(out.keys()),),
            )
            for doc_id, page_number, char_start, char_end in cur.fetchall():
                out[str(doc_id)]["pages"].append({"page_number": page_number, "char_start": char_start, "char_end": char_end})
    return out


def delete_document(document_id: str, user_id: str) -> bool:
    """Delete a document and everything derived from it (chunks, embeddings,
    page map, session links — all cascade via FK), plus the original file in
    R2. Returns False if the document doesn't exist or isn't owned by
    user_id.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, r2_url FROM documents WHERE id = %s::uuid", (document_id,))
            row = cur.fetchone()
            if not row or str(row[0]) != user_id:
                return False
            r2_url = row[1]
            cur.execute("DELETE FROM documents WHERE id = %s::uuid", (document_id,))
        conn.commit()

    if r2_url:
        try:
            from app.services.storage import delete_file_from_r2
            delete_file_from_r2(r2_url)
        except Exception:
            # The DB row is already gone — don't fail the delete over a
            # best-effort storage cleanup; delete_file_from_r2 already logs
            # internally on failure.
            pass
    return True


def mark_document_failed(document_id: str, reason: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE documents SET status = 'failed' WHERE id = %s::uuid",
                (document_id,),
            )
        conn.commit()


def attach_document_to_session(session_id: str, document_id: str) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO session_documents (session_id, document_id)
                VALUES (%s::uuid, %s::uuid)
                ON CONFLICT DO NOTHING
                """,
                (session_id, document_id),
            )
        conn.commit()


def get_session_document_ids(session_id: str) -> list[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT document_id FROM session_documents WHERE session_id = %s::uuid",
                (session_id,),
            )
            return [str(r[0]) for r in cur.fetchall()]
