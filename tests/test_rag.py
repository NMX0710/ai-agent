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
    """Set required environment variables for tests."""
    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-api-key"}):
        yield


@pytest.fixture
def mock_documents():
    """Create mock recipe documents."""
    return [
        Document(
            page_content="Lean chicken breast cutting recipe: boiled chicken breast with broccoli, high protein and low fat",
            metadata={"source": "recipe1.md"},
        ),
        Document(
            page_content="Tomato egg noodles: a classic home-style dish, nutritious and easy to make",
            metadata={"source": "recipe2.md"},
        ),
        Document(
            page_content="Steamed sea bass: low fat and high protein, suitable for fat-loss periods",
            metadata={"source": "recipe3.md"},
        ),
        Document(
            page_content="Cucumber salad: refreshing, very low calorie",
            metadata={"source": "recipe4.md"},
        ),
    ]


@pytest.fixture
def mock_rag_pipeline(mock_documents):
    """Create a mocked RAG pipeline with simple keyword-based retrieval."""
    mock_pipeline = Mock()

    def mock_retrieve(state, top_k=4):
        question = state.get("question", "")
        relevant_docs = []

        # Simple keyword matching logic
        for doc in mock_documents:
            if any(keyword in question for keyword in ["fat loss", "chicken", "low fat"]):
                if any(k in doc.page_content.lower() for k in ["chicken", "fat", "low"]):
                    relevant_docs.append(doc)
            elif "tomato" in question.lower():
                if "tomato" in doc.page_content.lower():
                    relevant_docs.append(doc)

        # If nothing matches, return top_k defaults
        if not relevant_docs:
            relevant_docs = mock_documents[:top_k]

        return {"context": relevant_docs[:top_k]}

    mock_pipeline.retrieve = mock_retrieve
    return mock_pipeline


class TestRecipeAppRAGIntegration:

    @patch("app.recipe_app.RecipeAppRAGPipeline")
    @patch("app.recipe_app.DashScopeEmbeddings")
    @patch("app.recipe_app.ChatTongyi")
    def test_init_with_rag(self, mock_tongyi, mock_embeddings, mock_rag_class, mock_env):
        """Verify RecipeApp initializes the RAG pipeline correctly."""
        mock_model_instance = Mock()
        mock_tongyi.return_value = mock_model_instance

        mock_embedding_instance = Mock()
        mock_embeddings.return_value = mock_embedding_instance

        mock_rag_instance = Mock()
        mock_rag_class.return_value = mock_rag_instance

        app = RecipeApp()

        mock_embeddings.assert_called_once_with(
            model="text-embedding-v1",
            dashscope_api_key="test-api-key",
        )
        mock_rag_class.assert_called_once_with(mock_embedding_instance, mock_model_instance)
        assert app.rag_pipeline == mock_rag_instance

    @patch("app.recipe_app.RecipeAppRAGPipeline")
    @patch("app.recipe_app.DashScopeEmbeddings")
    @patch("app.recipe_app.ChatTongyi")
    def test_graph_includes_retrieve_node(self, mock_tongyi, mock_embeddings, mock_rag_class, mock_env):
        """Verify the graph contains the retrieve node."""
        app = RecipeApp()

        nodes = app.graph.nodes
        assert "retrieve" in nodes
        assert "model" in nodes

        # Note: edge validation may require adjustments depending on LangGraph version.

    @patch("app.recipe_app.RecipeAppRAGPipeline")
    @patch("app.recipe_app.DashScopeEmbeddings")
    @patch("app.recipe_app.ChatTongyi")
    def test_retrieve_context_node(self, mock_tongyi, mock_embeddings, mock_rag_class, mock_env, mock_documents):
        """Test the retrieve-context node behavior."""
        mock_rag_instance = Mock()
        mock_rag_instance.retrieve.return_value = {"context": mock_documents[:2]}
        mock_rag_class.return_value = mock_rag_instance

        app = RecipeApp()

        # Case 1: state includes "question"
        state = {
            "question": "fat loss meal recommendations",
            "messages": [],
            "context": [],
            "answer": None,
        }

        result = app._retrieve_context(state)

        mock_rag_instance.retrieve.assert_called_once_with(state, top_k=4)
        assert "context" in result
        assert len(result["context"]) == 2

        # Case 2: "question" missing; extract from messages
        state_no_question = {
            "messages": [HumanMessage(content="Any low-fat recipe recommendations?")],
            "context": [],
            "answer": None,
        }

        result2 = app._retrieve_context(state_no_question)

        assert mock_rag_instance.retrieve.call_count == 2
        called_state = mock_rag_instance.retrieve.call_args[0][0]
        assert called_state["question"] == "Any low-fat recipe recommendations?"

    @patch("app.recipe_app.RecipeAppRAGPipeline")
    @patch("app.recipe_app.DashScopeEmbeddings")
    @patch("app.recipe_app.ChatTongyi")
    def test_call_model_with_context(self, mock_tongyi, mock_embeddings, mock_rag_class, mock_env, mock_documents):
        """Verify the model node incorporates retrieved context into the prompt."""
        mock_response = Mock()
        mock_response.content = "Based on your fat-loss goal, I recommend boiled chicken breast with broccoli."

        mock_model_instance = Mock()
        mock_model_instance.invoke.return_value = mock_response

        mock_tongyi.return_value = mock_model_instance

        app = RecipeApp()

        state = {
            "question": "fat loss meal recommendations",
            "messages": [HumanMessage(content="fat loss meal recommendations")],
            "context": mock_documents[:2],
            "answer": None,
        }

        result = app._call_model(state)

        mock_model_instance.invoke.assert_called_once()
        invoke_args = mock_model_instance.invoke.call_args[0][0]

        prompt_str = str(invoke_args)
        assert "Lean chicken breast cutting recipe" in prompt_str
        assert "Tomato egg noodles" in prompt_str

        assert "messages" in result
        assert "answer" in result
        assert len(result["messages"]) == 2
        assert result["messages"][-1] == mock_response
        assert result["answer"] == "Based on your fat-loss goal, I recommend boiled chicken breast with broccoli."

    @pytest.mark.asyncio
    @patch("app.recipe_app.RecipeAppRAGPipeline")
    @patch("app.recipe_app.DashScopeEmbeddings")
    @patch("app.recipe_app.ChatTongyi")
    async def test_chat_with_rag_integration(
        self,
        mock_tongyi,
        mock_embeddings,
        mock_rag_class,
        mock_env,
        mock_rag_pipeline,
    ):
        """Verify the full chat flow includes RAG behavior."""
        mock_response = Mock()
        mock_response.content = (
            "Based on your fat-loss goal, I recommend boiled chicken breast with broccoli. "
            "It is high protein and low fat, perfect for a cutting phase."
        )

        mock_model_instance = Mock()
        mock_model_instance.invoke.return_value = mock_response

        mock_tongyi.return_value = mock_model_instance
        mock_rag_class.return_value = mock_rag_pipeline

        app = RecipeApp()

        expected_output = {
            "question": "recommend some fat-loss recipes",
            "messages": [
                HumanMessage(content="recommend some fat-loss recipes"),
                mock_response,
            ],
            "context": mock_rag_pipeline.retrieve({"question": "recommend some fat-loss recipes"}, top_k=4)["context"],
            "answer": (
                "Based on your fat-loss goal, I recommend boiled chicken breast with broccoli. "
                "It is high protein and low fat, perfect for a cutting phase."
            ),
        }

        with patch.object(app.graph, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            mock_ainvoke.return_value = expected_output

            result = await app.chat("test-chat-id", "recommend some fat-loss recipes")

            mock_ainvoke.assert_called_once()
            call_args = mock_ainvoke.call_args

            input_state = call_args[0][0]
            assert input_state["question"] == "recommend some fat-loss recipes"
            assert len(input_state["messages"]) == 1
            assert isinstance(input_state["messages"][0], HumanMessage)
            assert "context" in input_state
            assert input_state["answer"] is None

            assert "high protein" in result.lower() or "protein" in result.lower()
            assert "chicken" in result.lower()

    @patch("app.recipe_app.RecipeAppRAGPipeline")
    @patch("app.recipe_app.DashScopeEmbeddings")
    @patch("app.recipe_app.ChatTongyi")
    def test_rag_retrieve_different_queries(
        self,
        mock_tongyi,
        mock_embeddings,
        mock_rag_class,
        mock_env,
        mock_rag_pipeline,
    ):
        """Test RAG retrieval results for different query types."""
        mock_rag_class.return_value = mock_rag_pipeline

        app = RecipeApp()

        # Fat-loss related query
        state1 = {"question": "What should I eat during a cutting phase?"}
        result1 = app._retrieve_context(state1)
        assert any(
            ("low" in doc.page_content.lower()) or ("fat" in doc.page_content.lower())
            for doc in result1["context"]
        )

        # Tomato related query
        state2 = {"question": "How to cook tomatoes?", "messages": []}
        result2 = app._retrieve_context(state2)
        assert any("tomato" in doc.page_content.lower() for doc in result2["context"])

        # Unrelated query (should return some default docs)
        state3 = {"question": "How is the weather today?", "messages": []}
        result3 = app._retrieve_context(state3)
        assert len(result3["context"]) > 0


@pytest.mark.asyncio
async def test_end_to_end_rag_flow(mock_env):
    """End-to-end test: verify retrieved context changes the model output."""
    with patch("app.recipe_app.ChatTongyi") as mock_tongyi, \
         patch("app.recipe_app.DashScopeEmbeddings") as mock_embeddings, \
         patch("app.recipe_app.RecipeAppRAGPipeline") as mock_rag_class:

        response_with_context = Mock(content="Based on the recipe library, try boiled chicken breast — a dedicated cutting recipe.")
        response_without_context = Mock(content="I can recommend some recipes for you.")

        mock_model_instance = Mock()

        def invoke_with_context_check(prompt):
            prompt_str = str(prompt)
            if "Below is relevant recipe information" in prompt_str and "Lean chicken breast cutting recipe" in prompt_str:
                return response_with_context
            else:
                return response_without_context

        mock_model_instance.invoke = invoke_with_context_check
        mock_tongyi.return_value = mock_model_instance

        mock_rag_instance = Mock()
        mock_rag_instance.retrieve.return_value = {
            "context": [Document(page_content="Lean chicken breast cutting recipe: boiled chicken breast with broccoli, high protein and low fat")]
        }
        mock_rag_class.return_value = mock_rag_instance

        app = RecipeApp()

        with patch.object(app.graph, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
            async def mock_graph_execution(state, config):
                retrieve_result = app._retrieve_context(state)
                state.update(retrieve_result)

                model_result = app._call_model(state)
                state.update(model_result)

                return state

            mock_ainvoke.side_effect = mock_graph_execution

            result = await app.chat("test-id", "recommend cutting recipes")

            assert "recipe library" in result.lower()
            assert "chicken" in result.lower()
            assert "cutting" in result.lower()
