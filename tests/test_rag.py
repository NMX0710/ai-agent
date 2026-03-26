import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from app.recipe_app import RecipeApp


@pytest.fixture
def mock_env():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
        yield


@pytest.fixture
def mock_documents():
    return [
        Document(page_content="Chicken breast fat-loss recipe: poached chicken breast with broccoli.", metadata={"source": "recipe1.md"}),
        Document(page_content="Tomato egg noodles: quick, practical, and easy for weeknights.", metadata={"source": "recipe2.md"}),
        Document(page_content="Steamed sea bass: high-protein and lower-fat.", metadata={"source": "recipe3.md"}),
        Document(page_content="Cucumber salad: refreshing and low calorie.", metadata={"source": "recipe4.md"}),
    ]


class TestRecipeAppRAGIntegration:
    @patch('app.recipe_app.RecipeAppRAGPipeline')
    @patch('app.recipe_app.OpenAIEmbeddings')
    @patch('app.recipe_app.ChatOpenAI')
    def test_init_with_rag(self, mock_chat_openai, mock_embeddings, mock_rag_class, mock_env):
        mock_model_instance = Mock()
        mock_chat_openai.return_value = mock_model_instance

        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance

        mock_rag_instance = Mock()
        mock_rag_class.return_value = mock_rag_instance

        app = RecipeApp()

        mock_embeddings.assert_called_once_with(
            model="text-embedding-3-small",
            api_key="test-api-key",
        )
        mock_rag_class.assert_called_once_with(mock_embedding_instance, mock_model_instance)
        assert app.rag_pipeline == mock_rag_instance

    @patch('app.recipe_app.RecipeAppRAGPipeline')
    @patch('app.recipe_app.OpenAIEmbeddings')
    @patch('app.recipe_app.ChatOpenAI')
    def test_graph_includes_retrieve_node(self, mock_chat_openai, mock_embeddings, mock_rag_class, mock_env):
        app = RecipeApp()
        nodes = app.graph.nodes
        assert "retrieve" in nodes
        assert "model" in nodes

    @patch('app.recipe_app.RecipeAppRAGPipeline')
    @patch('app.recipe_app.OpenAIEmbeddings')
    @patch('app.recipe_app.ChatOpenAI')
    def test_build_agent_messages_uses_english_prompt(self, mock_chat_openai, mock_embeddings, mock_rag_class, mock_env, mock_documents):
        mock_rag_class.return_value = Mock()
        app = RecipeApp()
        messages_with_ctx, rag_context = app._build_agent_messages([HumanMessage(content="Suggest a fat-loss dinner.")], mock_documents[:2])
        system_message = messages_with_ctx[0]

        assert "Always answer in English" in system_message.content
        assert "Retrieved recipe context" in system_message.content
        assert "Chicken breast fat-loss recipe" in rag_context

    @patch('app.recipe_app.RecipeAppRAGPipeline')
    @patch('app.recipe_app.OpenAIEmbeddings')
    @patch('app.recipe_app.ChatOpenAI')
    def test_generate_dpo_record_exports_trace_fields(self, mock_chat_openai, mock_embeddings, mock_rag_class, mock_env, mock_documents):
        mock_rag_instance = Mock()
        mock_rag_instance.retrieve.return_value = {"context": mock_documents[:2]}
        mock_rag_class.return_value = mock_rag_instance

        returned_messages = [
            HumanMessage(content="Suggest a fast tomato dinner."),
            AIMessage(content="Try tomato egg noodles with a side salad."),
        ]
        mock_executor = Mock()
        mock_executor.invoke.return_value = {"messages": returned_messages}

        app = RecipeApp()
        app.agent_executor = mock_executor

        record = app.generate_dpo_record(
            sample_id="sample-1",
            user_input="Suggest a fast tomato dinner.",
            metadata={"scenario_type": "quick_meal"},
        )

        assert record["sample_id"] == "sample-1"
        assert record["user_input"] == "Suggest a fast tomato dinner."
        assert len(record["retrieved_context"]) == 2
        assert record["chosen"] == "Try tomato egg noodles with a side salad."
        assert record["agent_input"]["messages"][0]["role"] == "system"
        assert "messages_before_agent_call" in record
        assert "returned_messages" in record
        assert record["model_provider"] == "openai"


@pytest.mark.asyncio
@patch('app.recipe_app.RecipeAppRAGPipeline')
@patch('app.recipe_app.OpenAIEmbeddings')
@patch('app.recipe_app.ChatOpenAI')
async def test_chat_uses_graph_output(mock_chat_openai, mock_embeddings, mock_rag_class, mock_env):
    app = RecipeApp()
    expected_output = {
        "question": "Suggest a fat-loss dinner.",
        "messages": [
            HumanMessage(content="Suggest a fat-loss dinner."),
            AIMessage(content="Try grilled chicken with broccoli and roasted potatoes."),
        ],
        "context": [],
        "answer": "Try grilled chicken with broccoli and roasted potatoes.",
    }

    with patch.object(app.graph, 'ainvoke', new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = expected_output
        result = await app.chat("test-chat-id", "Suggest a fat-loss dinner.")

    assert "grilled chicken" in result
