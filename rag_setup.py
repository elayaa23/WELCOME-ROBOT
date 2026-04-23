import os
import json
import hashlib
from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = "documents"
DB_DIR = "chroma_db"
HASH_CACHE = ".indexed_files.json"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_hash_cache():
    if os.path.exists(HASH_CACHE):
        with open(HASH_CACHE) as f:
            return json.load(f)
    return {}

def save_hash_cache(cache):
    with open(HASH_CACHE, "w") as f:
        json.dump(cache, f, indent=2)

def main():
    print("\n🔍 RAG Setup — building vector database\n")
    os.makedirs(DOCS_DIR, exist_ok=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    cache = load_hash_cache()
    skip_hashes = set(cache.keys())
    all_chunks = []
    new_hashes = {}

    for file in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, file)
        ext = Path(file).suffix.lower()
        if ext not in (".pdf", ".docx", ".txt"):
            continue
        fhash = file_hash(path)
        if fhash in skip_hashes:
            print(f"  ⏭  Skipping (already indexed): {file}")
            continue
        print(f"  📄 Loading: {file}")
        try:
            if ext == ".pdf":
                docs = PyPDFLoader(path).load()
            elif ext == ".docx":
                docs = Docx2txtLoader(path).load()
            else:
                docs = TextLoader(path, encoding="utf-8").load()
        except Exception as e:
            print(f"     ⚠️  Failed: {e}")
            continue
        for doc in docs:
            doc.metadata["source_file"] = file
        chunks = splitter.split_documents(docs)
        print(f"     → {len(chunks)} chunks")
        all_chunks.extend(chunks)
        new_hashes[fhash] = file

    if not all_chunks:
        print("\n✅ Nothing new to index.")
        return
    print(f"\n⚙️  Embedding {len(all_chunks)} chunks...")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    db.add_documents(all_chunks)
    cache.update(new_hashes)
    save_hash_cache(cache)
    total = db._collection.count()
    print(f"✅ Done! Database total: {total} chunks\n")

if __name__ == "__main__":
    main()
