import logging
import os
from pathlib import Path
from typing import List, TypedDict, Dict, Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore


class RecipeAppState(TypedDict, total=False):
    question: str
    context: List[Document]          # Retrieved chunks from the RAG retriever
    messages: List[BaseMessage]      # Conversation history managed by LangGraph
    memory: List[str]                # Retrieved long-term user memory snippets
    answer: Optional[str]            # Final answer produced by the pipeline

    plan: Optional[Dict[str, Any]]

class RecipeAppRAGPipeline:
    def __init__(self, embeddings: Embeddings, model: BaseChatModel):
        """
        Initialize the pipeline:
          - Locate and validate the local document folder
          - Build an in-memory vector index
          - Attach the LLM instance
        """
        folder_path = Path(__file__).parent.parent.parent / "resources" / "documents"
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid document folder path: {folder_path}")

        self.vector_store = self.indexing(folder_path, embeddings)

        if model is None:
            raise ValueError("The 'model' argument must not be None. Please pass a valid LLM instance from RecipeApp.")
        self.model = model

    @staticmethod
    def indexing(folder_path: Path, embeddings: Embeddings) -> InMemoryVectorStore:
        """
        Load local Markdown documents and build a vector store index.

        Steps:
          1) Load all Markdown files under the specified folder
          2) (Optional) Split large documents into smaller chunks
          3) Embed and store documents in an in-memory vector store
        """
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid folder path: {folder_path}")

        all_documents = load_markdown_docs(folder_path)
        logging.info("Loaded %d document chunks in total.", len(all_documents))

        # Optional: split documents if each Markdown file is too large.
        # from langchain_text_splitters import RecursiveCharacterTextSplitter
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # all_documents = text_splitter.split_documents(all_documents)

        vector_store = InMemoryVectorStore(embeddings)
        _ = vector_store.add_documents(documents=all_documents)
        return vector_store

    def retrieve(self, state: RecipeAppState, top_k: int = 4):
        """
        Perform semantic retrieval using the user's question.

        Args:
            state: LangGraph state containing the "question" field
            top_k: Number of most similar chunks to retrieve

        Returns:
            A partial state update: {"context": List[Document]}
            (Returns an empty list on failure rather than None.)
        """
        question = state["question"]
        logging.info("[RAG Retrieve] Query: %s", question)

        try:
            retrieved_docs = self.vector_store.similarity_search(question, k=top_k)
            logging.info("[RAG Retrieve] Retrieved %d relevant chunks.", len(retrieved_docs))
            return {"context": retrieved_docs}
        except Exception as e:
            logging.exception("[RAG Retrieve] Retrieval failed.")
            return {"context": []}


def load_markdown_docs(folder_path: Path) -> list[Document]:
    """
    Load all *.md files under the given folder as LangChain Document objects.

    Each Document stores:
      - page_content: raw Markdown text
      - metadata["source"]: file path for traceability
    """
    docs: list[Document] = []
    for file in folder_path.glob("*.md"):
        text = file.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": str(file)}))
    return docs
