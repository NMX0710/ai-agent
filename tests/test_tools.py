import os
import sys
import asyncio
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent if Path(__file__).parent.name == "tests" else Path(__file__).parent
sys.path.append(str(project_root))

print("=== 开始测试工具 ===")


# 测试1: Web搜索工具
def test_web_search():
    print("\n🔍 测试Web搜索工具...")
    try:
        from app.tools.web_search_tool import web_search

        print("✅ 从web_search_tool成功导入web_search")

        query = "程序员鱼皮编程导航 codefather.cn"
        print(f"🔍 搜索查询: {query}")

        result = web_search.invoke(query)

        print(f"📄 搜索结果类型: {type(result)}")
        if isinstance(result, dict) and 'results' in result:
            print(f"📊 找到 {len(result['results'])} 条结果")
            if result['results']:
                first_result = result['results'][0]
                print(f"第一条结果: {first_result.get('title', 'No title')}")
        else:
            print(f"📊 搜索结果: {str(result)[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Web搜索测试失败: {e}")
        return False


# 测试2: 网页抓取工具
def test_web_scraping():
    print("\n🕷️ 测试网页抓取工具...")
    try:
        from app.tools.web_scraping_tool import scrape_web_page

        print("✅ 从web_scraping_tool成功导入scrape_web_page")

        # 测试抓取一个简单的网页
        test_url = "https://httpbin.org/html"  # 这是一个测试网站，返回简单HTML
        print(f"🌐 抓取URL: {test_url}")

        result = scrape_web_page.invoke(test_url)

        print(f"📄 抓取结果类型: {type(result)}")
        print(f"📄 内容长度: {len(result)} 字符")

        # 检查是否包含HTML标签
        if "<html" in result.lower() and "</html>" in result.lower():
            print("✅ 成功抓取到HTML内容")
            # 显示前200个字符
            print(f"📊 内容预览: {result[:200]}...")
        elif "error" in result.lower():
            print(f"⚠️ 抓取出现错误: {result}")
        else:
            print(f"📊 抓取结果: {result[:200]}...")

        return True

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保已安装: pip install requests beautifulsoup4")
        return False
    except Exception as e:
        print(f"❌ 网页抓取测试失败: {e}")
        return False


# 测试3: 抓取编程导航网站
def test_scrape_codefather():
    print("\n🐟 测试抓取编程导航网站...")
    try:
        from app.tools.web_scraping_tool import scrape_web_page

        test_url = "https://www.codefather.cn"
        print(f"🌐 抓取URL: {test_url}")

        result = scrape_web_page.invoke(test_url)

        print(f"📄 抓取结果类型: {type(result)}")
        print(f"📄 内容长度: {len(result)} 字符")

        # 检查是否包含编程导航相关内容
        if "编程导航" in result or "codefather" in result.lower():
            print("✅ 成功抓取到编程导航网站内容")
        elif "error" in result.lower():
            print(f"⚠️ 抓取出现错误: {result}")
        else:
            print("✅ 抓取完成")

        print(f"📊 内容预览: {result[:300]}...")
        return True

    except Exception as e:
        print(f"❌ 抓取编程导航失败: {e}")
        return False


# 测试4: 终端操作工具
def test_terminal_operations():
    print("\n💻 测试终端操作工具...")
    try:
        from app.tools.terminal_operation_tool import execute_terminal_command

        print("✅ 从terminal_operation_tool成功导入execute_terminal_command")

        # 测试基础命令
        test_commands = [
            ("pwd", "获取当前目录"),
            ("whoami", "获取当前用户"),
            ("date", "获取当前时间"),
            ("echo 'Hello from terminal tool!'", "测试echo命令"),
            ("ls -la | head -5", "测试管道命令")
        ]

        success_count = 0

        for command, description in test_commands:
            print(f"\n🔨 {description}: {command}")

            try:
                result = execute_terminal_command.invoke(command)

                if "error" in result.lower():
                    print(f"⚠️ 命令执行有问题: {result[:100]}...")
                else:
                    print(f"✅ 执行成功")
                    print(f"📄 输出: {result[:100]}...")
                    success_count += 1

            except Exception as cmd_error:
                print(f"❌ 命令执行失败: {cmd_error}")

        print(f"\n📊 终端测试汇总: {success_count}/{len(test_commands)} 个命令成功执行")
        return success_count > 0

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 终端操作测试失败: {e}")
        return False


# 测试6: 资源下载工具
def test_resource_download():
    print("\n📥 测试资源下载工具...")
    try:
        from app.tools.resource_download_tool import download_resource

        print("✅ 从resource_download_tool成功导入download_resource")

        # 测试图片下载
        test_urls = [
            ("https://en.wikipedia.org/wiki/File:Chiikawa_volume_1_cover.jpg", "test_image.png", "下载测试图片"),
            ("https://jsonplaceholder.typicode.com/posts/1", "sample.json", "下载JSON数据"),
            ("https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore", "Python.gitignore",
             "下载文本文件")
        ]

        success_count = 0

        for test_url, file_name, description in test_urls:
            print(f"\n📋 {description}")
            print(f"🌐 下载URL: {test_url}")
            print(f"📁 文件名: {file_name}")

            try:
                result = download_resource.invoke({"url": test_url, "file_name": file_name})

                print(f"📄 下载结果: {result}")

                # 检查是否下载成功
                if "successfully" in result.lower():
                    print("✅ 文件下载成功")
                elif "error" in result.lower():
                    print(f"⚠️ 下载出现错误: {result}")
                else:
                    print(f"📊 下载结果: {result}")
                    success_count += 1

            except Exception as download_error:
                print(f"❌ 下载失败: {download_error}")

        print(f"\n📊 下载测试汇总: {success_count}/{len(test_urls)} 个文件成功下载")
        return success_count > 0

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保已安装: pip install requests")
        return False
    except Exception as e:
        print(f"❌ 资源下载测试失败: {e}")
        return False

# 测试7: PDF 生成工具
def test_pdf_generation():
    print("\n📄 测试PDF生成工具...")
    try:
        from app.tools.pdf_generation_tool import generate_pdf

        print("✅ 从pdf_generation_tool成功导入generate_pdf")

        file_name = "test_output.pdf"
        content = "Hello, this is a test PDF generated by AI Agent!"

        print(f"📁 文件名: {file_name}")
        print(f"📝 内容: {content}")

        result = generate_pdf.invoke({"file_name": file_name, "content": content})

        print(f"📄 返回结果: {result}")

        # 检查文件是否生成成功
        if "successfully" in str(result).lower() and file_name in str(result):
            print("✅ PDF文件生成成功")
            return True
        else:
            print("⚠️ PDF生成结果异常")
            return False

    except Exception as e:
        print(f"❌ PDF生成测试失败: {e}")
        return False

# 添加项目路径
project_root = Path(__file__).parent.parent if Path(__file__).parent.name == "tests" else Path(__file__).parent
sys.path.append(str(project_root))

from app.recipe_app import RecipeApp

# 要测试的英文 message（每条都应触发某个工具）
TEST_MESSAGES = [
    # Web Search Tool
    "Use the web search tool to find trending low-carb meal plans for weight loss. Please perform the search.",

    # Web Scraping Tool
    "Use the web scraping tool to extract content from https://codefather.cn about healthy recipes.",

    # Resource Download Tool
    "Please download a sample meal plan PDF from https://example.com/sample.pdf and save it as diet_plan.pdf.",

    # Terminal Operation Tool
    "Run the terminal command: echo 'Daily calorie intake: 2100 kcal'.",

    # File Operation Tool
    "Save the following preferences into a file: I avoid gluten and prefer plant-based protein.",

    # PDF Generation Tool
    "Generate a PDF titled 'Weekly Vegan Meal Plan' with content: Breakfast - oats, Lunch - tofu salad, Dinner - lentil soup."
]



async def tool_message_runner(message: str) -> bool:
    from app.recipe_app import RecipeApp  # 避免循环引用
    app = RecipeApp()

    print(f"\n🔍 Testing tool with message:\n   \"{message}\"")

    try:
        result = await app.chat(chat_id="debug-chat", message=message)
        print(f"🧠 Model Response:\n   {result}")

        # 工具触发关键词 → 工具名称映射
        tool_keywords = {
            "search": "🔍 Web Search Tool",
            "scrape": "🕷️ Web Scraping Tool",
            "download": "📥 Resource Download Tool",
            "terminal": "💻 Terminal Operation Tool",
            "shell": "💻 Terminal Operation Tool",
            "file": "📝 File Operation Tool",
            "pdf": "📄 PDF Generation Tool"
        }

        # 工具命中标记
        triggered_tools = [name for keyword, name in tool_keywords.items() if keyword in result.lower()]
        if triggered_tools:
            print("✅ Triggered Tools:", "、".join(triggered_tools))
            return True
        else:
            print("❌ No tools triggered based on response content.")
            return False

    except Exception as e:
        print(f"❌ Error while testing message: {e}")
        return False



def run_all_tool_tests():
    print("=== 开始测试工具调用能力 ===")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = []
    for msg in TEST_MESSAGES:
        result = loop.run_until_complete(tool_message_runner(msg))
        results.append((msg, result))

    print("\n📊 测试结果汇总:")
    for i, (msg, success) in enumerate(results):
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{i+1}. {msg[:40]}... -> {status}")

    passed = sum(1 for _, success in results if success)
    print(f"\n总计: {passed}/{len(TEST_MESSAGES)} 个测试通过")
    print("=== 测试结束 ===")

# 运行所有测试
if __name__ == "__main__":
    # results = []
    #
    # # 运行测试
    # results.append(("Web搜索", test_web_search()))
    # results.append(("网页抓取", test_web_scraping()))
    # results.append(("抓取编程导航", test_scrape_codefather()))
    # results.append(("终端操作", test_terminal_operations()))
    #
    # # 显示测试结果汇总
    # print("\n" + "=" * 50)
    # print("📊 测试结果汇总:")
    # for test_name, success in results:
    #     status = "✅ 成功" if success else "❌ 失败"
    #     print(f"{test_name}: {status}")
    #
    # successful_tests = sum(1 for _, success in results if success)
    # print(f"\n总计: {successful_tests}/{len(results)} 个测试通过")
    # print("=== 测试结束 ===")
    run_all_tool_tests()