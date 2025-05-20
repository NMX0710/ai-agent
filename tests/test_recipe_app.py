import pytest
import os
from app.recipe_app import RecipeApp

@pytest.mark.asyncio
class TestRecipeAppMemory:
    """
    测试 RecipeApp 是否支持多轮对话记忆功能。
    """

    @pytest.fixture(scope="class")
    def recipe_app(self):
        return RecipeApp()

    @pytest.mark.asyncio
    async def test_memory_across_turns(self, recipe_app):
        """
        测试模型是否能记住用户提到的食材或背景信息。
        """
        chat_id = "test-memory-user-001"

        # 第一次对话：告诉模型我喜欢吃鸡胸肉
        message1 = "你好，我是Elva，我喜欢吃鸡胸肉。"
        response1 = await recipe_app.chat(chat_id, message1)
        assert response1 is not None and isinstance(response1, str)
        print("第一轮回答：", response1)

        # 第二次：请求推荐减脂菜谱
        message2 = "我最近在减脂，晚上想吃清淡点的，可以推荐菜吗？"
        response2 = await recipe_app.chat(chat_id, message2)
        assert response2 is not None and isinstance(response2, str)
        print("第二轮回答：", response2)

        # 第三次：回忆我说过的偏好
        message3 = "我刚才说喜欢吃什么来着？"
        response3 = await recipe_app.chat(chat_id, message3)
        assert response3 is not None and isinstance(response3, str)
        print("第三轮回答：", response3)

        # 检查模型是否正确记住了“鸡胸肉”
        assert "鸡胸肉" in response3 or "鸡" in response3
