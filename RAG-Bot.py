import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:65532"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:65532"

from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

pdf_dir = Path("凡人修仙传")
documents = []

for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    doc = Document(
        page_content=full_text,
        metadata={"source":pdf_path.name}
    )
    documents.append(doc)

spliter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    separators=["\n\n", "\n", "。", "，", " ", ""]
)

chunks = spliter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
)

vectorsotre = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="fanren_novel",
    persist_directory="./chroma_db",
)

queries = [
    "谁是主角？",
    "主角叫什么名字？",
    "主角家住哪里？"
]

for q in queries:
    results = vectorsotre.similarity_search(q,k=4)
    for j,doc in enumerate(results):
        print(f"  [{j+1}] {doc.page_content[:100]}...")