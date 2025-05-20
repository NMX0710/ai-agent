import pytest
from app.recipe_app import RecipeApp

@pytest.fixture(scope="module")
def recipe_app():
    """
    初始化 RecipeApp，供多个测试共享
    """
    return RecipeApp()

@pytest.mark.asyncio
async def test_memory_across_turns(recipe_app):
    """
    测试模型是否能记住用户提到的食材或背景信息。
    """
    chat_id = "test-memory-user-001"

    message1 = "你好，我是Elva，我喜欢吃鸡胸肉。"
    response1 = await recipe_app.chat(chat_id, message1)
    assert response1 and isinstance(response1, str)
    print("第一轮回答：", response1)

    message2 = "我最近在减脂，晚上想吃清淡点的，可以推荐菜吗？"
    response2 = await recipe_app.chat(chat_id, message2)
    assert response2 and isinstance(response2, str)
    print("第二轮回答：", response2)

    message3 = "我刚才说喜欢吃什么来着？"
    response3 = await recipe_app.chat(chat_id, message3)
    assert response3 and isinstance(response3, str)
    print("第三轮回答：", response3)

    assert "鸡胸肉" in response3 or "鸡" in response3

#TODO：为什么就拿到log了？
@pytest.mark.asyncio
async def test_logger_output(recipe_app, caplog):
    """
    测试 logger 是否正确记录用户输入和模型输出
    """
    chat_id = "log-test-user"
    test_input = "我喜欢吃西兰花，帮我推荐一个菜谱吧～"

    with caplog.at_level("INFO"):
        response = await recipe_app.chat(chat_id, test_input)

    # 打印日志供人工核查
    for record in caplog.records:
        print(f"[log] {record.message}")

    # 验证日志记录了输入
    assert any("用户输入" in record.message and "西兰花" in record.message for record in caplog.records)
    # 验证日志记录了输出
    assert any("模型回复" in record.message for record in caplog.records)

    assert response and isinstance(response, str)
