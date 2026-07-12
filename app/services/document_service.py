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
                INSERT INTO documents (user_id, filename, content)
                VALUES (%s::uuid, %s, %s)
                RETURNING id
                """,
                (user_id, filename, text),
            )
            document_id: str = str(cur.fetchone()[0])

            # ── 3. Insert chunks + embeddings ────────────────────────────────
            chunk_metadatas: list[dict] = []
            global_chunk_idx = 0
            
            for p in hierarchical_chunks:
                # 3a. Insert Parent chunk (no embedding)
                cur.execute(
                    """
                    INSERT INTO document_chunks (document_id, chunk_index, content)
                    VALUES (%s::uuid, %s, %s)
                    RETURNING id
                    """,
                    (document_id, global_chunk_idx, p["parent"]),
                )
                parent_id: str = str(cur.fetchone()[0])
                global_chunk_idx += 1
                
                # 3b. Insert Child chunks (with embedding)
                for child_text in p["children"]:
                    cur.execute(
                        """
                        INSERT INTO document_chunks (document_id, parent_chunk_id, chunk_index, content)
                        VALUES (%s::uuid, %s::uuid, %s, %s)
                        RETURNING id
                        """,
                        (document_id, parent_id, global_chunk_idx, child_text),
                    )
                    child_id: str = str(cur.fetchone()[0])
                    global_chunk_idx += 1
                    
                    # 3c. Insert embedding for child
                    vector = vectors[vector_idx]
                    vector_idx += 1
                    
                    cur.execute(
                        """
                        INSERT INTO embeddings (chunk_id, embedding)
                        VALUES (%s::uuid, %s::vector)
                        """,
                        (child_id, str(vector)),
                    )

                    chunk_metadatas.append({
                        "chunk_id": child_id,
                        "document_id": document_id,
                        "parent_chunk_id": parent_id,
                        "chunk_index": global_chunk_idx - 1,
                        "filename": filename,
                        "user_id": user_id,
                    })
                    
                if progress_callback:
                    # 50% to 100% for DB insertion
                    # Calculate progress based on vectors inserted
                    progress_callback(50 + int((vector_idx / total_chunks) * 50))

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
