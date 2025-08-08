import pytest
import asyncio
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from app.recipe_app import RecipeApp, RecipeReport
from app.rag.recipe_app_rag_pipeline import RecipeAppState


@pytest.fixture
def mock_env():
    """设置测试环境变量"""
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-api-key"}):
        yield


@pytest.fixture
def mock_documents():
    """创建模拟的文档数据"""
    return [
        Document(page_content="鸡胸肉减脂食谱：水煮鸡胸肉配西兰花，高蛋白低脂肪", metadata={"source": "recipe1.md"}),
        Document(page_content="番茄鸡蛋面：经典家常菜，营养丰富，制作简单", metadata={"source": "recipe2.md"}),
        Document(page_content="清蒸鲈鱼：低脂高蛋白，适合减脂期食用", metadata={"source": "recipe3.md"}),
        Document(page_content="凉拌黄瓜：爽口开胃，热量极低", metadata={"source": "recipe4.md"})
    ]


@pytest.fixture
def mock_rag_pipeline(mock_documents):
    """创建模拟的 RAG Pipeline"""
    mock_pipeline = Mock()

    def mock_retrieve(state, top_k=4):
        question = state.get("question", "")
        # 简单的关键词匹配逻辑
        relevant_docs = []
        for doc in mock_documents:
            if any(keyword in question for keyword in ["减脂", "鸡", "低脂"]):
                if "鸡" in doc.page_content or "减脂" in doc.page_content or "低脂" in doc.page_content:
                    relevant_docs.append(doc)
            elif "番茄" in question:
                if "番茄" in doc.page_content:
                    relevant_docs.append(doc)

        # 如果没有匹配的，返回前 top_k 个
        if not relevant_docs:
            relevant_docs = mock_documents[:top_k]

        return {"context": relevant_docs[:top_k]}

    mock_pipeline.retrieve = mock_retrieve
    return mock_pipeline


class TestRecipeAppRAGIntegration:

    @patch('app.recipe_app.RecipeAppRAGPipeline')
    @patch('app.recipe_app.DashScopeEmbeddings')
    @patch('app.recipe_app.ChatTongyi')
    def test_init_with_rag(self, mock_tongyi, mock_embeddings, mock_rag_class, mock_env):
        """测试 RecipeApp 初始化时是否正确创建 RAG pipeline"""
        # 设置模拟返回值
        mock_model_instance = Mock()
        mock_tongyi.return_value = mock_model_instance

        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance

        mock_rag_instance = Mock()
        mock_rag_class.return_value = mock_rag_instance

        # 创建 RecipeApp 实例
        app = RecipeApp()

        # 验证 RAG pipeline 是否被正确初始化
        mock_embeddings.assert_called_once_with(
            model="text-embedding-v1",
            dashscope_api_key="test-api-key"
        )
        mock_rag_class.assert_called_once_with(mock_embedding_instance, mock_model_instance)
        assert app.rag_pipeline == mock_rag_instance

    @patch('app.recipe_app.RecipeAppRAGPipeline')
    @patch('app.recipe_app.DashScopeEmbeddings')
    @patch('app.recipe_app.ChatTongyi')
    def test_graph_includes_retrieve_node(self, mock_tongyi, mock_embeddings, mock_rag_class, mock_env):
        """测试 graph 是否包含 retrieve 节点"""
        app = RecipeApp()

        # 检查 graph 的节点
        nodes = app.graph.nodes
        assert "retrieve" in nodes
        assert "model" in nodes

        # 验证节点顺序（通过检查边）
        # 注意：这里的具体实现可能需要根据 langgraph 的 API 调整

    @patch('app.recipe_app.RecipeAppRAGPipeline')
    @patch('app.recipe_app.DashScopeEmbeddings')
    @patch('app.recipe_app.ChatTongyi')
    def test_retrieve_context_node(self, mock_tongyi, mock_embeddings, mock_rag_class, mock_env, mock_documents):
        """测试 _retrieve_context 节点功能"""
        # 设置模拟
        mock_rag_instance = Mock()
        mock_rag_instance.retrieve.return_value = {"context": mock_documents[:2]}
        mock_rag_class.return_value = mock_rag_instance

        app = RecipeApp()

        # 测试 case 1: state 中有 question
        state = {
            "question": "减脂餐推荐",
            "messages": [],
            "context": [],
            "answer": None
        }

        result = app._retrieve_context(state)

        # 验证调用
        mock_rag_instance.retrieve.assert_called_once_with(state, top_k=4)
        assert "context" in result
        assert len(result["context"]) == 2

        # 测试 case 2: state 中没有 question，从 messages 中提取
        state_no_question = {
            "messages": [HumanMessage(content="有什么低脂食谱推荐吗？")],
            "context": [],
            "answer": None
        }

        result2 = app._retrieve_context(state_no_question)

        # 验证 question 被正确提取
        assert mock_rag_instance.retrieve.call_count == 2
        called_state = mock_rag_instance.retrieve.call_args[0][0]
        assert called_state["question"] == "有什么低脂食谱推荐吗？"

    @patch('app.recipe_app.RecipeAppRAGPipeline')
    @patch('app.recipe_app.DashScopeEmbeddings')
    @patch('app.recipe_app.ChatTongyi')
    def test_call_model_with_context(self, mock_tongyi, mock_embeddings, mock_rag_class, mock_env, mock_documents):
        """测试 _call_model 节点是否正确使用 context"""
        # 创建模拟的模型响应
        mock_response = Mock()
        mock_response.content = "根据您的减脂需求，我推荐水煮鸡胸肉配西兰花。"

        # 创建模拟的模型实例
        mock_model_instance = Mock()
        mock_model_instance.invoke.return_value = mock_response

        # 设置模拟
        mock_tongyi.return_value = mock_model_instance

        app = RecipeApp()

        # 准备包含 context 的 state
        state = {
            "question": "减脂餐推荐",
            "messages": [HumanMessage(content="减脂餐推荐")],
            "context": mock_documents[:2],
            "answer": None
        }

        result = app._call_model(state)

        # 验证模型调用时包含了 context
        mock_model_instance.invoke.assert_called_once()
        invoke_args = mock_model_instance.invoke.call_args[0][0]

        # 检查 prompt 中是否包含 context 内容
        prompt_str = str(invoke_args)
        assert "鸡胸肉减脂食谱" in prompt_str
        assert "番茄鸡蛋面" in prompt_str

        # 验证返回结果
        assert "messages" in result
        assert "answer" in result
        assert len(result["messages"]) == 2  # 原始消息 + AI 回复
        assert result["messages"][-1] == mock_response  # AI 回复
        assert result["answer"] == "根据您的减脂需求，我推荐水煮鸡胸肉配西兰花。"

    @pytest.mark.asyncio
    @patch('app.recipe_app.RecipeAppRAGPipeline')
    @patch('app.recipe_app.DashScopeEmbeddings')
    @patch('app.recipe_app.ChatTongyi')
    async def test_chat_with_rag_integration(self, mock_tongyi, mock_embeddings, mock_rag_class, mock_env,
                                             mock_rag_pipeline):
        """测试完整的 chat 流程是否包含 RAG"""
        # 创建模拟的模型响应
        mock_response = Mock()
        mock_response.content = "根据您的减脂需求，我推荐水煮鸡胸肉配西兰花，这道菜高蛋白低脂肪，非常适合减脂期食用。"

        # 创建模拟的模型实例
        mock_model_instance = Mock()
        mock_model_instance.invoke.return_value = mock_response

        # 设置模拟
        mock_tongyi.return_value = mock_model_instance
        mock_rag_class.return_value = mock_rag_pipeline

        app = RecipeApp()

        # 模拟 graph 的 ainvoke 方法
        expected_output = {
            "question": "推荐一些减脂食谱",
            "messages": [
                HumanMessage(content="推荐一些减脂食谱"),
                mock_response
            ],
            "context": mock_rag_pipeline.retrieve({"question": "推荐一些减脂食谱"}, top_k=4)["context"],
            "answer": "根据您的减脂需求，我推荐水煮鸡胸肉配西兰花，这道菜高蛋白低脂肪，非常适合减脂期食用。"
        }

        with patch.object(app.graph, 'ainvoke', new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = expected_output

            # 执行 chat
            result = await app.chat("test-chat-id", "推荐一些减脂食谱")

            # 验证
            mock_ainvoke.assert_called_once()
            call_args = mock_ainvoke.call_args

            # 检查传入的 state
            input_state = call_args[0][0]
            assert input_state["question"] == "推荐一些减脂食谱"
            assert len(input_state["messages"]) == 1
            assert isinstance(input_state["messages"][0], HumanMessage)
            assert "context" in input_state
            assert input_state["answer"] is None

            # 检查返回结果
            assert "减脂" in result
            assert "鸡胸肉" in result

    @patch('app.recipe_app.RecipeAppRAGPipeline')
    @patch('app.recipe_app.DashScopeEmbeddings')
    @patch('app.recipe_app.ChatTongyi')
    def test_rag_retrieve_different_queries(self, mock_tongyi, mock_embeddings, mock_rag_class, mock_env,
                                            mock_rag_pipeline):
        """测试不同查询的 RAG 检索结果"""
        mock_rag_class.return_value = mock_rag_pipeline

        app = RecipeApp()

        # 测试减脂相关查询
        state1 = {"question": "减脂期间吃什么好"}
        result1 = app._retrieve_context(state1)
        assert any("减脂" in doc.page_content or "低脂" in doc.page_content
                   for doc in result1["context"])

        # 测试番茄相关查询
        state2 = {"question": "番茄怎么做", "messages": []}
        result2 = app._retrieve_context(state2)
        assert any("番茄" in doc.page_content for doc in result2["context"])

        # 测试无关查询（应返回默认文档）
        state3 = {"question": "今天天气怎么样", "messages": []}
        result3 = app._retrieve_context(state3)
        assert len(result3["context"]) > 0  # 应该返回一些默认文档


@pytest.mark.asyncio
async def test_end_to_end_rag_flow(mock_env):
    """端到端测试：验证 RAG 是否影响最终输出"""
    with patch('app.recipe_app.ChatTongyi') as mock_tongyi, \
            patch('app.recipe_app.DashScopeEmbeddings') as mock_embeddings, \
            patch('app.recipe_app.RecipeAppRAGPipeline') as mock_rag_class:

        # 创建一个能区分是否使用了 RAG 的模拟模型
        response_with_context = Mock(content="基于菜谱库，我推荐您尝试水煮鸡胸肉，这是专门的减脂食谱。")
        response_without_context = Mock(content="我可以推荐一些食谱给您。")

        mock_model_instance = Mock()

        def invoke_with_context_check(prompt):
            prompt_str = str(prompt)
            if "以下是相关的菜谱信息" in prompt_str and "鸡胸肉减脂食谱" in prompt_str:
                return response_with_context
            else:
                return response_without_context

        mock_model_instance.invoke = invoke_with_context_check
        mock_tongyi.return_value = mock_model_instance

        # 设置 RAG pipeline
        mock_rag_instance = Mock()
        mock_rag_instance.retrieve.return_value = {
            "context": [Document(page_content="鸡胸肉减脂食谱：水煮鸡胸肉配西兰花，高蛋白低脂肪")]
        }
        mock_rag_class.return_value = mock_rag_instance

        app = RecipeApp()

        # 模拟完整的 graph 执行
        with patch.object(app.graph, 'ainvoke', new_callable=AsyncMock) as mock_ainvoke:
            async def mock_graph_execution(state, config):
                # 模拟 retrieve 节点
                retrieve_result = app._retrieve_context(state)
                state.update(retrieve_result)

                # 模拟 model 节点
                model_result = app._call_model(state)
                state.update(model_result)

                return state

            mock_ainvoke.side_effect = mock_graph_execution

            # 执行 chat
            result = await app.chat("test-id", "推荐减脂食谱")

            # 验证结果包含了基于 RAG 的内容
            assert "基于菜谱库" in result
            assert "水煮鸡胸肉" in result
            assert "专门的减脂食谱" in result