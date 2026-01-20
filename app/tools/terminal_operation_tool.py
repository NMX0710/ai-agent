import subprocess
import shlex
from langchain_core.tools import tool


@tool(description="Execute a command in the terminal")
def execute_terminal_command(command: str) -> str:
    """
    Execute a shell command in the local terminal and return its output.

    Args:
        command: The command string to execute

    Returns:
        The command output (stdout and stderr) or an error message
    """
    try:
        # On macOS / Linux, commands are executed via the default shell (bash/zsh).
        # Since this environment is macOS, no Windows-specific handling is needed.
        #
        # NOTE:
        # - shell=True is intentionally enabled to allow pipes, redirection, etc.
        # - A timeout is enforced to prevent long-running or stuck commands.

        result = subprocess.run(
            command,
            shell=True,              # Allow shell features such as pipes and redirection
            capture_output=True,     # Capture stdout and stderr
            text=True,               # Return output as text instead of bytes
            timeout=30,              # Abort if execution exceeds 30 seconds
            cwd=None                 # Use the current working directory
        )

        output_lines = []

        # Append standard output if present
        if result.stdout:
            output_lines.append(result.stdout.rstrip())

        # Append standard error output if present
        if result.stderr:
            output_lines.append(f"STDERR: {result.stderr.rstrip()}")

        # If the command failed (non-zero exit code), include it explicitly
        if result.returncode != 0:
            output_lines.append(
                f"Command execution failed with exit code: {result.returncode}"
            )

        return (
            "\n".join(output_lines)
            if output_lines
            else "Command executed successfully (no output)"
        )

    except subprocess.TimeoutExpired:
        return "Error executing command: command timed out after 30 seconds"
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e}"
    except Exception as e:
        return f"Error executing command: {str(e)}"
