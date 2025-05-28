import logging
import os
from pathlib import Path
from typing import List, TypedDict, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from app.recipe_app import RecipeAppState


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

        # 初始化 RAG 专用的提示模板
        self.rag_template = ChatPromptTemplate.from_messages([
            ("system",
             "你是一位擅长中餐和西餐的厨师专家，拥有丰富的烹饪经验和营养知识。"
             "你的任务是根据用户提供的饮食偏好、口味、食材、健康需求等信息，推荐合适的菜谱或建议。"
             "如果用户是减脂人群，推荐低热量高蛋白的清淡菜式；如果用户想改善家庭饮食质量，可推荐易做营养的家常菜；"
             "若用户提到特定食材（如鸡肉、番茄等），请结合食材特点推荐合适做法。"
             "\n\n以下是相关的菜谱信息，请基于这些信息回答用户问题：\n{context}"),
            MessagesPlaceholder(variable_name="messages"),
        ])

    @staticmethod
    def indexing(folder_path: Path, embeddings: Embeddings) -> InMemoryVectorStore:
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
            logging.info(f"加载文档: {file.name}, 共 {len(docs)} 个元素")

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

    def retrieve(self, state: RecipeAppState, top_k: int = 4) -> Dict[str, List[Document]]:
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
            return {"context": []}

    def generate(self, state: RecipeAppState) -> Dict[str, str]:
        """
        基于状态生成回答
        """
        question = state["question"]
        context_docs = state.get("context", [])
        messages = state.get("messages", [])

        logging.info(f"[RAG Generate] 生成答案，问题: {question}")
        logging.info(f"[RAG Generate] 上下文文档数量: {len(context_docs)}")

        try:
            # 构建上下文文本
            docs_content = "\n\n".join(doc.page_content for doc in context_docs)
            if not docs_content.strip():
                docs_content = "暂无相关菜谱信息"

            # 构建消息列表 - 包含历史消息和当前问题
            current_messages = messages + [HumanMessage(content=question)]

            # 使用模板生成提示
            prompt = self.rag_template.invoke({
                "context": docs_content,
                "messages": current_messages
            })

            # 调用模型生成回答
            response = self.model.invoke(prompt)
            answer = response.content

            logging.info(f"[RAG Generate] 生成答案成功: {answer[:100]}...")
            return {"answer": answer}

        except Exception as e:
            logging.error(f"[RAG Generate] 生成答案失败: {e}")
            return {"answer": "抱歉，我暂时无法处理您的请求，请稍后再试。"}

    def rag_query(self, state: RecipeAppState, top_k: int = 4) -> RecipeAppState:
        """
        完整的 RAG 查询流程：检索 + 生成
        返回更新后的完整状态
        """
        logging.info(f"[RAG Query] 开始处理: {state['question']}")

        # 创建状态副本
        updated_state = state.copy()

        try:
            # 第一步：检索
            retrieve_result = self.retrieve(state, top_k)
            updated_state["context"] = retrieve_result["context"]

            # 第二步：生成
            generate_result = self.generate(updated_state)
            updated_state["answer"] = generate_result["answer"]

            logging.info(f"[RAG Query] 处理完成，检索到 {len(updated_state['context'])} 个文档")
            return updated_state

        except Exception as e:
            logging.error(f"[RAG Query] 处理失败: {e}")
            updated_state["context"] = []
            updated_state["answer"] = "抱歉，我暂时无法处理您的请求，请稍后再试。"
            return updated_state