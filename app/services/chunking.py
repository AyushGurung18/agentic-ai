from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int, chunk_overlap: int):
    """
    Splits text into chunks using natural boundaries (newlines, spaces).
    Legacy method for simple chunking, returns list of strings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)

def hierarchical_chunk_text(text: str, parent_chunk_size: int = 1500, parent_chunk_overlap: int = 150, child_chunk_size: int = 300, child_chunk_overlap: int = 50):
    """
    Splits text into large parent chunks, and then splits each parent into smaller child chunks.
    Returns a list of dicts:
    [
        {
            "parent": "Large text...",
            "children": ["small part 1...", "small part 2..."]
        },
        ...
    ]
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    
    parent_texts = parent_splitter.split_text(text)
    
    results = []
    for p_text in parent_texts:
        c_texts = child_splitter.split_text(p_text)
        results.append({
            "parent": p_text,
            "children": c_texts
        })
        
    return results
