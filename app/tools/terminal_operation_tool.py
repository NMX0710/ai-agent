import subprocess
from langchain_core.tools import tool


@tool(description="Execute a command in the terminal")
def execute_terminal_command(command: str) -> str:
    """Execute a shell command and return stdout/stderr text."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=None,
        )

        output_lines = []
        if result.stdout:
            output_lines.append(result.stdout.rstrip())
        if result.stderr:
            output_lines.append(f"STDERR: {result.stderr.rstrip()}")
        if result.returncode != 0:
            output_lines.append(f"Command execution failed with exit code: {result.returncode}")

        return "\n".join(output_lines) if output_lines else "Command executed successfully (no output)"
    except subprocess.TimeoutExpired:
        return "Error executing command: Command timed out after 30 seconds"
    except subprocess.CalledProcessError as exc:
        return f"Error executing command: {exc}"
    except Exception as exc:
        return f"Error executing command: {exc}"
