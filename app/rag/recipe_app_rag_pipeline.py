import os
from pathlib import Path
from typing import List, TypedDict

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
class RecipeAppRAGPipeline:
    def __init__(self, embeddings: Embeddings):
        """
        初始化：加载文档，创建向量索引，初始化 LLM 与 Prompt
        """
        folder_path = Path(__file__).parent.parent.parent / "resources" / "documents"
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"路径无效: {folder_path}")
        self.vector_store = self.indexing(folder_path, embeddings)

    @staticmethod
    def indexing(folder_path: Path, embeddings: Embeddings = None) -> InMemoryVectorStore:
        """
        Loading Documents
        批量加载本地 Markdown 文档并返回 LangChain Document 对象列表
        """
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"指定路径无效: {folder_path}")

        all_documents = []
        for file in folder_path.glob("*.md"):
            loader = UnstructuredMarkdownLoader(str(file), mode="elements")
            # 因为我们的mode用的是element，md file里面每一个元素都会是一个Document
            docs = loader.load()
            all_documents.extend(docs)

        """
        Split Documents(Optional)
        Split documents if the size of each document is too large
        """

        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        #all_splits = text_splitter.split_documents(docs)

        """
        Storing Documents
        """
        vector_store = InMemoryVectorStore(embeddings)
        _ = vector_store.add_documents(documents=all_documents)
        return vector_store



    # TODO：完成retrieve，然后传递到recipe_app里使用
    def retrieve(self, question: str, top_k: int = 4) -> List[Document]:
        """
        根据问题进行语义检索，返回相关文档块
        """
        return self.vector_store.similarity_search(question, k=top_k)

    #Generate logic is in recipe_app








