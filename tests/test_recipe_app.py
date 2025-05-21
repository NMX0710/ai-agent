import pytest
from app.recipe_app import RecipeApp
from app.recipe_app import RecipeReport

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

# caplog：pytest 内置的 日志捕捉对象，可以捕捉 logging 输出
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



@pytest.mark.asyncio
def test_generate_report_structure(recipe_app):
    """
    测试 generate_report() 返回结构化的 RecipeReport，
    并且各字段类型和合理性（只检查 title 和 suggestions）。
    """
    chat_id = "report-test-user"
    report =  recipe_app.generate_report(
        chat_id,
        "我想吃低脂高蛋白的晚餐，用鸡胸肉和西兰花做简单的菜"
    )

    #打印看一下
    print(report)

