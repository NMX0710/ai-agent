import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore



# 自动拼出绝对路径：以当前 Python 文件为基准
base_path = Path(__file__).parent.parent.parent
folder_path_default = base_path / "resources" / "documents"

def indexing(folder_path: Path = folder_path_default, mode: str = "elements", embeddings: Embeddings = None) -> InMemoryVectorStore:
    """
    Loading Documents
    批量加载本地 Markdown 文档并返回 LangChain Document 对象列表
    """
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"指定路径无效: {folder_path}")

    all_documents = []
    for file in folder_path.glob("*.md"):
        loader = UnstructuredMarkdownLoader(str(file), mode=mode)
        # 因为我们的mode用的是element，md file里面每一个元素都会是一个document
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







