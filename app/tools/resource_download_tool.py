import requests
import os
from pathlib import Path
from urllib.parse import urlparse
from langchain_core.tools import tool
from ..settings import FILE_SAVE_DIR


@tool(description="Download a resource from a given URL")
def download_resource(url: str, file_name: str) -> str:
    """
    从给定URL下载资源

    Args:
        url: 要下载资源的URL
        file_name: 保存下载资源的文件名

    Returns:
        成功信息或错误信息
    """
    try:
        download_dir = FILE_SAVE_DIR
        download_dir.mkdir(parents=True, exist_ok=True)

        # 构造完整文件路径
        file_path = download_dir / file_name

        # 设置请求头，模拟浏览器
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # 发送HTTP请求下载文件
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()  # 检查HTTP错误

        # 写入文件
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # 过滤掉keep-alive的空块
                    file.write(chunk)

        return f"Resource downloaded successfully to: {file_path}"

    except requests.exceptions.RequestException as e:
        return f"Error downloading resource: {str(e)}"
    except IOError as e:
        return f"Error writing file: {str(e)}"
    except Exception as e:
        return f"Error downloading resource: {str(e)}"

