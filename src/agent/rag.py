"""RAG sobre regulatorios brasileiros usando ChromaDB + sentence-transformers."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter

KB_DIR = Path("data/knowledge_base")
CHROMA_DIR = Path("data/chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index() -> Chroma:
    """Indexa todos os .md de data/knowledge_base/."""
    if not KB_DIR.exists() or not list(KB_DIR.glob("*.md")):
        raise RuntimeError(f"Sem documentos em {KB_DIR}")

    splitter = MarkdownTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = []
    for md_file in KB_DIR.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        for chunk in splitter.split_text(text):
            docs.append({
                "page_content": chunk,
                "metadata": {"source": md_file.name},
            })

    from langchain_core.documents import Document
    documents = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs]

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vs = Chroma.from_documents(
        documents=documents,
        embedding=_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_name="regulatorios_br",
    )
    print(f">>> Indexados {len(documents)} chunks de {len(list(KB_DIR.glob('*.md')))} arquivos")
    return vs


@lru_cache(maxsize=1)
def _load_index() -> Chroma:
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=_embeddings(),
        collection_name="regulatorios_br",
    )


def search(query: str, k: int = 3) -> list[dict]:
    vs = _load_index()
    results = vs.similarity_search_with_score(query, k=k)
    return [
        {
            "source": doc.metadata.get("source"),
            "score": round(float(score), 4),
            "snippet": doc.page_content,
        }
        for doc, score in results
    ]


if __name__ == "__main__":
    print(">>> Indexando knowledge base...")
    build_index()
    print("\n>>> Teste de busca:")
    for r in search("Qual a regra para revisao humana em decisoes automatizadas?"):
        print(f"\n[{r['source']} | dist={r['score']}]")
        print(r["snippet"][:200])
