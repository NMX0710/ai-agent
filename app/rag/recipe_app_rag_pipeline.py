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


class RecipeAppState(TypedDict):
    question: str
    context: List[Document]  # RAG 检索到的内容
    messages: List[BaseMessage]  # 历史消息记录（LangGraph 自带）
    answer: Optional[str]  # 最终答案



class RecipeAppRAGPipeline:
    def __init__(self, embeddings: Embeddings, model: BaseChatModel):
        """
        初始化：加载文档，创建向量索引，初始化 LLM 与 Prompt
        """
        folder_path = Path(__file__).parent.parent.parent / "resources" / "documents"
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"路径无效: {folder_path}")

        self.vector_store = self.indexing(folder_path, embeddings)

        if model is None:
            raise ValueError("model 参数不能为空，请从 RecipeApp 传入模型实例")
        self.model = model

    @staticmethod
    def indexing(folder_path: Path, embeddings: Embeddings) -> InMemoryVectorStore:
        """
        Loading Documents
        批量加载本地 Markdown 文档并返回 LangChain Document 对象列表
        """
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"指定路径无效: {folder_path}")

        all_documents = load_markdown_docs(folder_path)

        logging.info(f"总共加载了 {len(all_documents)} 个文档片段")

        """
        Split Documents(Optional) 
        Split documents if the size of each document is too large
        """
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # all_splits = text_splitter.split_documents(docs)

        """
        Storing Documents
        """
        vector_store = InMemoryVectorStore(embeddings)
        _ = vector_store.add_documents(documents=all_documents)
        return vector_store

    def retrieve(self, state: RecipeAppState, top_k: int = 4):
        """
        根据问题进行语义检索，返回相关文档块
        """
        question = state["question"]
        logging.info(f"[RAG Retrieve] 检索问题: {question}")

        try:
            retrieved_docs = self.vector_store.similarity_search(question, k=top_k)
            logging.info(f"[RAG Retrieve] 检索到 {len(retrieved_docs)} 个相关文档")
            return {"context": retrieved_docs}
        except Exception as e:
            logging.error(f"[RAG Retrieve] 检索失败: {e}")
            return {"context": []}  # 返回空列表而不是 None


def load_markdown_docs(folder_path: Path) -> list[Document]:
    docs: list[Document] = []
    for file in folder_path.glob("*.md"):
        text = file.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": str(file)}))
    return docs