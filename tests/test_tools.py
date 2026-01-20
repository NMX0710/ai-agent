import os
import sys
import asyncio
from pathlib import Path

# ------------------------------------------------------------
# Add project root to sys.path so "app.*" imports work
# ------------------------------------------------------------
project_root = (
    Path(__file__).parent.parent
    if Path(__file__).parent.name == "tests"
    else Path(__file__).parent
)
sys.path.append(str(project_root))

print("=== Starting Tool Tests ===")


# ------------------------------------------------------------
# Test 1: Web Search Tool
# ------------------------------------------------------------
def test_web_search() -> bool:
    print("\n🔍 Testing Web Search tool...")
    try:
        from app.tools.web_search_tool import web_search

        print("✅ Successfully imported web_search from web_search_tool")

        query = "codefather.cn programming navigation"
        print(f"🔍 Search query: {query}")

        result = web_search.invoke(query)

        print(f"📄 Result type: {type(result)}")
        if isinstance(result, dict) and "results" in result:
            print(f"📊 Found {len(result['results'])} results")
            if result["results"]:
                first_result = result["results"][0]
                print(f"First result title: {first_result.get('title', 'No title')}")
        else:
            print(f"📊 Result preview: {str(result)[:200]}...")

        return True

    except Exception as e:
        print(f"❌ Web Search test failed: {e}")
        return False


# ------------------------------------------------------------
# Test 2: Web Scraping Tool
# ------------------------------------------------------------
def test_web_scraping() -> bool:
    print("\n🕷️ Testing Web Scraping tool...")
    try:
        from app.tools.web_scraping_tool import scrape_web_page

        print("✅ Successfully imported scrape_web_page from web_scraping_tool")

        # Fetch a simple HTML page from httpbin (great for testing)
        test_url = "https://httpbin.org/html"
        print(f"🌐 Scraping URL: {test_url}")

        result = scrape_web_page.invoke(test_url)

        print(f"📄 Result type: {type(result)}")
        print(f"📄 Content length: {len(result)} characters")

        if "<html" in result.lower() and "</html>" in result.lower():
            print("✅ Successfully scraped HTML content")
            print(f"📊 Preview: {result[:200]}...")
        elif "error" in result.lower():
            print(f"⚠️ Scraping returned an error: {result}")
        else:
            print(f"📊 Preview: {result[:200]}...")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Please ensure dependencies are installed: pip install requests beautifulsoup4")
        return False
    except Exception as e:
        print(f"❌ Web Scraping test failed: {e}")
        return False


# ------------------------------------------------------------
# Test 3: Scrape CodeFather website (optional, may be blocked)
# ------------------------------------------------------------
def test_scrape_codefather() -> bool:
    print("\n🐟 Testing scraping CodeFather website...")
    try:
        from app.tools.web_scraping_tool import scrape_web_page

        test_url = "https://www.codefather.cn"
        print(f"🌐 Scraping URL: {test_url}")

        result = scrape_web_page.invoke(test_url)

        print(f"📄 Result type: {type(result)}")
        print(f"📄 Content length: {len(result)} characters")

        if "codefather" in result.lower():
            print("✅ Successfully scraped CodeFather content")
        elif "error" in result.lower():
            print(f"⚠️ Scraping returned an error: {result}")
        else:
            print("✅ Scrape completed (content may be dynamic)")

        print(f"📊 Preview: {result[:300]}...")
        return True

    except Exception as e:
        print(f"❌ Failed to scrape CodeFather: {e}")
        return False


# ------------------------------------------------------------
# Test 4: Terminal Operation Tool
# ------------------------------------------------------------
def test_terminal_operations() -> bool:
    print("\n💻 Testing Terminal Operation tool...")
    try:
        from app.tools.terminal_operation_tool import execute_terminal_command

        print("✅ Successfully imported execute_terminal_command from terminal_operation_tool")

        test_commands = [
            ("pwd", "Get current directory"),
            ("whoami", "Get current user"),
            ("date", "Get current time"),
            ("echo 'Hello from terminal tool!'", "Test echo"),
            ("ls -la | head -5", "Test piping"),
        ]

        success_count = 0

        for command, description in test_commands:
            print(f"\n🔨 {description}: {command}")
            try:
                result = execute_terminal_command.invoke(command)

                if "error" in result.lower():
                    print(f"⚠️ Command returned an error: {result[:120]}...")
                else:
                    print("✅ Command executed successfully")
                    print(f"📄 Output preview: {result[:120]}...")
                    success_count += 1

            except Exception as cmd_error:
                print(f"❌ Command execution failed: {cmd_error}")

        print(f"\n📊 Terminal test summary: {success_count}/{len(test_commands)} commands succeeded")
        return success_count > 0

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Terminal operation test failed: {e}")
        return False


# ------------------------------------------------------------
# Test 5: Resource Download Tool
# ------------------------------------------------------------
def test_resource_download() -> bool:
    print("\n📥 Testing Resource Download tool...")
    try:
        from app.tools.resource_download_tool import download_resource

        print("✅ Successfully imported download_resource from resource_download_tool")

        test_urls = [
            (
                "https://jsonplaceholder.typicode.com/posts/1",
                "sample.json",
                "Download a JSON sample",
            ),
            (
                "https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore",
                "Python.gitignore",
                "Download a text file",
            ),
        ]

        success_count = 0

        for test_url, file_name, description in test_urls:
            print(f"\n📋 {description}")
            print(f"🌐 URL: {test_url}")
            print(f"📁 File name: {file_name}")

            try:
                # NOTE: This tool expects a dict payload based on your implementation
                result = download_resource.invoke({"url": test_url, "file_name": file_name})
                print(f"📄 Result: {result}")

                if "successfully" in str(result).lower():
                    print("✅ Download succeeded")
                    success_count += 1
                elif "error" in str(result).lower():
                    print(f"⚠️ Download error: {result}")
                else:
                    print(f"ℹ️ Unexpected response: {result}")

            except Exception as download_error:
                print(f"❌ Download failed: {download_error}")

        print(f"\n📊 Download test summary: {success_count}/{len(test_urls)} files downloaded")
        return success_count > 0

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Please ensure dependency is installed: pip install requests")
        return False
    except Exception as e:
        print(f"❌ Resource download test failed: {e}")
        return False


# ------------------------------------------------------------
# Test 6: PDF Generation Tool
# ------------------------------------------------------------
def test_pdf_generation() -> bool:
    print("\n📄 Testing PDF Generation tool...")
    try:
        from app.tools.pdf_generation_tool import generate_pdf

        print("✅ Successfully imported generate_pdf from pdf_generation_tool")

        file_name = "test_output.pdf"
        content = "Hello, this is a test PDF generated by the agent tool!"

        print(f"📁 File name: {file_name}")
        print(f"📝 Content: {content}")

        result = generate_pdf.invoke({"file_name": file_name, "content": content})
        print(f"📄 Result: {result}")

        if "successfully" in str(result).lower() and file_name in str(result):
            print("✅ PDF generated successfully")
            return True

        print("⚠️ PDF generation returned an unexpected result")
        return False

    except Exception as e:
        print(f"❌ PDF generation test failed: {e}")
        return False


# ------------------------------------------------------------
# Agent-level tool triggering tests (via RecipeApp chat)
# ------------------------------------------------------------
from app.recipe_app import RecipeApp

# Each message below is intended to encourage the agent to use a specific tool.
TEST_MESSAGES = [
    # Web Search Tool
    "Use the web search tool to find trending low-carb meal plans for weight loss. Please perform the search.",

    # Web Scraping Tool
    "Use the web scraping tool to extract content from https://codefather.cn about healthy recipes.",

    # Resource Download Tool (note: example.com is not a real file source; may fail by design)
    "Please download a sample meal plan PDF from https://example.com/sample.pdf and save it as diet_plan.pdf.",

    # Terminal Operation Tool
    "Run the terminal command: echo 'Daily calorie intake: 2100 kcal'.",

    # File Operation Tool (depends on how your read/write tools are implemented)
    "Save the following preferences into a file: I avoid gluten and prefer plant-based protein.",

    # PDF Generation Tool
    "Generate a PDF titled 'Weekly Vegan Meal Plan' with content: Breakfast - oats, Lunch - tofu salad, Dinner - lentil soup.",
]


async def tool_message_runner(message: str) -> bool:
    # Lazy import to avoid potential circular imports in some setups
    app = RecipeApp()

    print(f'\n🔍 Testing agent tool usage with message:\n   "{message}"')

    try:
        result = await app.chat(chat_id="debug-chat", message=message)
        print(f"🧠 Model response:\n   {result}")

        # Heuristic keyword mapping: detect likely tool usage from the response text
        tool_keywords = {
            "search": "🔍 Web Search Tool",
            "scrape": "🕷️ Web Scraping Tool",
            "download": "📥 Resource Download Tool",
            "terminal": "💻 Terminal Operation Tool",
            "shell": "💻 Terminal Operation Tool",
            "file": "📝 File Operation Tool",
            "pdf": "📄 PDF Generation Tool",
        }

        triggered_tools = [
            name for keyword, name in tool_keywords.items()
            if keyword in str(result).lower()
        ]

        if triggered_tools:
            print("✅ Detected tool usage (heuristic):", " / ".join(triggered_tools))
            return True

        print("❌ No tool usage detected based on response text.")
        return False

    except Exception as e:
        print(f"❌ Error while testing message: {e}")
        return False


def run_all_tool_tests():
    print("=== Starting agent tool invocation tests ===")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    results = []
    for msg in TEST_MESSAGES:
        ok = loop.run_until_complete(tool_message_runner(msg))
        results.append((msg, ok))

    print("\n📊 Summary:")
    for i, (msg, success) in enumerate(results):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{i + 1}. {msg[:50]}... -> {status}")

    passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {passed}/{len(TEST_MESSAGES)} tests passed")
    print("=== Done ===")


if __name__ == "__main__":
    # You can optionally run the direct tool tests first:
    #
    # results = []
    # results.append(("Web Search", test_web_search()))
    # results.append(("Web Scraping", test_web_scraping()))
    # results.append(("Scrape CodeFather", test_scrape_codefather()))
    # results.append(("Terminal Operations", test_terminal_operations()))
    # results.append(("Resource Download", test_resource_download()))
    # results.append(("PDF Generation", test_pdf_generation()))
    #
    # print("\n" + "=" * 60)
    # print("📊 Direct tool test summary:")
    # for test_name, success in results:
    #     status = "✅ PASS" if success else "❌ FAIL"
    #     print(f"{test_name}: {status}")
    #
    # successful_tests = sum(1 for _, success in results if success)
    # print(f"\nTotal: {successful_tests}/{len(results)} tests passed")
    # print("=== Done ===")

    run_all_tool_tests()
