import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

class RecipeAppDocumentLoader:
    """
    批量加载本地 Markdown 文档并返回 LangChain Document 对象列表
    """
    def __init__(self, folder_path:str = None, mode:str = "elements"):
        """
        :param folder_path: Markdown 文档所在的文件夹路径
        :param mode: "single" 将整篇文档合并为一个 Document；
                     "elements" 会保留不同段落/结构元素为多个 Document。
        """
        if folder_path is None:
            # 自动拼出绝对路径：以当前 Python 文件为基准
            base_path = Path(__file__).parent.parent.parent  # 到项目根目录
            folder_path = base_path / "resources" / "documents"
        self.folder_path = Path(folder_path)
        self.mode = mode

    def load_md(self) -> List[Document]:
        if not self.folder_path.exists() or not self.folder_path.is_dir():
            raise ValueError(f"指定路径无效: {self.folder_path}")
        all_documents = []

        for file in self.folder_path.glob("*.md"):
            loader = UnstructuredMarkdownLoader(str(file), mode = self.mode)
            # 因为我们的mode用的是element，md file里面每一个元素都会是一个document
            docs = loader.load()
            all_documents.extend(docs)
        return all_documents

