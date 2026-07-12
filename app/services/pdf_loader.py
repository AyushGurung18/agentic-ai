import tempfile
import os
import pymupdf4llm

def extract_text(file):
    # 'file' is a SpooledTemporaryFile from FastAPI UploadFile
    # We need to save it to disk temporarily because pymupdf4llm needs a filename
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    try:
        # Generate markdown from the PDF using pymupdf4llm
        md_text = pymupdf4llm.to_markdown(tmp_file_path)
        print(md_text)
        return md_text
    finally:
        # Ensure we always clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)