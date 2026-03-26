import logging
from pathlib import Path
from typing import List, Optional, TypedDict

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.vectorstores import InMemoryVectorStore


class RecipeAppState(TypedDict):
    question: str
    context: List[Document]
    messages: List[BaseMessage]
    answer: Optional[str]


class RecipeAppRAGPipeline:
    def __init__(self, embeddings: Embeddings, model: BaseChatModel):
        folder_path = Path(__file__).parent.parent.parent / "resources" / "documents"
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid documents path: {folder_path}")

        self.vector_store = self.indexing(folder_path, embeddings)

        if model is None:
            raise ValueError("model must not be None")
        self.model = model

    @staticmethod
    def indexing(folder_path: Path, embeddings: Embeddings) -> InMemoryVectorStore:
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid documents path: {folder_path}")

        all_documents = []
        for file in folder_path.glob("*.md"):
            loader = UnstructuredMarkdownLoader(str(file), mode="elements")
            docs = loader.load()
            all_documents.extend(docs)
            logging.info(f"Loaded document: {file.name}, elements={len(docs)}")

        logging.info(f"Loaded {len(all_documents)} document chunks in total")

        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents=all_documents)
        return vector_store

    def retrieve(self, state: RecipeAppState, top_k: int = 4):
        question = state["question"]
        logging.info(f"[RAG Retrieve] question={question}")

        try:
            retrieved_docs = self.vector_store.similarity_search(question, k=top_k)
            logging.info(f"[RAG Retrieve] retrieved={len(retrieved_docs)}")
            return {"context": retrieved_docs}
        except Exception as exc:
            logging.error(f"[RAG Retrieve] failed: {exc}")
            return {"context": []}
