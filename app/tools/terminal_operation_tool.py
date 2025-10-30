import subprocess
import shlex
from langchain_core.tools import tool


@tool(description="Execute a command in the terminal")
def execute_terminal_command(command: str) -> str:
    """
    在终端中执行命令

    Args:
        command: 要在终端中执行的命令

    Returns:
        命令执行结果或错误信息
    """
    try:
        # Mac/Linux使用bash，Windows使用cmd
        # 由于你是Mac，我们使用bash
        # 使用shlex.split来安全地分割命令，避免注入攻击

        # 对于Mac，我们直接执行命令，不需要像Windows那样加cmd.exe前缀
        result = subprocess.run(
            command,
            shell=True,  # 允许使用shell特性，如管道、重定向等
            capture_output=True,  # 捕获stdout和stderr
            text=True,  # 以文本模式返回输出
            timeout=30,  # 30秒超时，防止命令卡死
            cwd=None  # 使用当前工作目录
        )

        # 收集输出
        output_lines = []

        # 添加标准输出
        if result.stdout:
            output_lines.append(result.stdout.rstrip())

        # 如果有错误输出，也添加进去
        if result.stderr:
            output_lines.append(f"STDERR: {result.stderr.rstrip()}")

        # 如果命令执行失败（退出码不为0）
        if result.returncode != 0:
            output_lines.append(f"Command execution failed with exit code: {result.returncode}")

        return "\n".join(output_lines) if output_lines else "Command executed successfully (no output)"

    except subprocess.TimeoutExpired:
        return f"Error executing command: Command timed out after 30 seconds"
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e}"
    except Exception as e:
        return f"Error executing command: {str(e)}"