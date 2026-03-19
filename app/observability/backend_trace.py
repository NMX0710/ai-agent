from __future__ import annotations

from typing import Any

from deepagents.backends.protocol import SandboxBackendProtocol

from app.observability.tracing import trace_log


def _preview_text(value: Any, limit: int = 300) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _is_memory_path(path: str | None) -> bool:
    return bool(path and path.startswith("/memories/"))


def _should_log_path(path: str | None) -> bool:
    return bool(path and (_is_memory_path(path) or path.startswith("/skills/")))


def _should_log_paths(paths: list[str] | None) -> bool:
    return any(_should_log_path(path) for path in (paths or []))


class TracingBackend(SandboxBackendProtocol):
    """Delegating backend wrapper that traces backend-level file operations."""

    def __init__(self, backend: Any):
        self._backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Reading file",
                {"file_path": file_path, "offset": offset, "limit": limit},
            )
        result = self._backend.read(file_path, offset=offset, limit=limit)
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Read file result",
                {"file_path": file_path, "content_preview": _preview_text(result)},
            )
        return result

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Reading file",
                {"file_path": file_path, "offset": offset, "limit": limit},
            )
        result = await self._backend.aread(file_path, offset=offset, limit=limit)
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Read file result",
                {"file_path": file_path, "content_preview": _preview_text(result)},
            )
        return result

    def write(self, file_path: str, content: str):
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Writing file",
                {"file_path": file_path, "content_preview": _preview_text(content)},
            )
        result = self._backend.write(file_path, content)
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Write file result",
                {"file_path": file_path, "result": str(result)},
            )
        return result

    async def awrite(self, file_path: str, content: str):
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Writing file",
                {"file_path": file_path, "content_preview": _preview_text(content)},
            )
        result = await self._backend.awrite(file_path, content)
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Write file result",
                {"file_path": file_path, "result": str(result)},
            )
        return result

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False):
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Editing file",
                {
                    "file_path": file_path,
                    "old_string_preview": _preview_text(old_string),
                    "new_string_preview": _preview_text(new_string),
                    "replace_all": replace_all,
                },
            )
        result = self._backend.edit(file_path, old_string, new_string, replace_all=replace_all)
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Edit file result",
                {"file_path": file_path, "result": str(result)},
            )
        return result

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False):
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Editing file",
                {
                    "file_path": file_path,
                    "old_string_preview": _preview_text(old_string),
                    "new_string_preview": _preview_text(new_string),
                    "replace_all": replace_all,
                },
            )
        result = await self._backend.aedit(file_path, old_string, new_string, replace_all=replace_all)
        if _should_log_path(file_path):
            trace_log(
                "AgentBackend",
                "Edit file result",
                {"file_path": file_path, "result": str(result)},
            )
        return result

    def download_files(self, paths: list[str]):
        if _should_log_paths(paths):
            trace_log("AgentBackend", "Downloading files", {"paths": paths})
        result = self._backend.download_files(paths)
        if _should_log_paths(paths):
            trace_log(
                "AgentBackend",
                "Download files result",
                {"count": len(result), "paths": paths},
            )
        return result

    async def adownload_files(self, paths: list[str]):
        if _should_log_paths(paths):
            trace_log("AgentBackend", "Downloading files", {"paths": paths})
        result = await self._backend.adownload_files(paths)
        if _should_log_paths(paths):
            trace_log(
                "AgentBackend",
                "Download files result",
                {"count": len(result), "paths": paths},
            )
        return result

    def upload_files(self, files: list[tuple[str, bytes]]):
        paths = [path for path, _ in files]
        if _should_log_paths(paths):
            trace_log("AgentBackend", "Uploading files", {"paths": paths})
        result = self._backend.upload_files(files)
        if _should_log_paths(paths):
            trace_log(
                "AgentBackend",
                "Upload files result",
                {"count": len(result), "paths": paths},
            )
        return result

    async def aupload_files(self, files: list[tuple[str, bytes]]):
        paths = [path for path, _ in files]
        if _should_log_paths(paths):
            trace_log("AgentBackend", "Uploading files", {"paths": paths})
        result = await self._backend.aupload_files(files)
        if _should_log_paths(paths):
            trace_log(
                "AgentBackend",
                "Upload files result",
                {"count": len(result), "paths": paths},
            )
        return result

    def ls_info(self, path: str):
        if _should_log_path(path):
            trace_log("AgentBackend", "Listing path", {"path": path})
        result = self._backend.ls_info(path)
        if _should_log_path(path):
            trace_log("AgentBackend", "List path result", {"path": path, "count": len(result)})
        return result

    async def als_info(self, path: str):
        if _should_log_path(path):
            trace_log("AgentBackend", "Listing path", {"path": path})
        result = await self._backend.als_info(path)
        if _should_log_path(path):
            trace_log("AgentBackend", "List path result", {"path": path, "count": len(result)})
        return result

    def glob_info(self, pattern: str, path: str = "/"):
        if _should_log_path(path):
            trace_log("AgentBackend", "Glob path", {"path": path, "pattern": pattern})
        result = self._backend.glob_info(pattern, path=path)
        if _should_log_path(path):
            trace_log("AgentBackend", "Glob path result", {"path": path, "pattern": pattern, "count": len(result)})
        return result

    async def aglob_info(self, pattern: str, path: str = "/"):
        if _should_log_path(path):
            trace_log("AgentBackend", "Glob path", {"path": path, "pattern": pattern})
        result = await self._backend.aglob_info(pattern, path=path)
        if _should_log_path(path):
            trace_log("AgentBackend", "Glob path result", {"path": path, "pattern": pattern, "count": len(result)})
        return result

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
        if _should_log_path(path):
            trace_log("AgentBackend", "Grep path", {"path": path, "pattern": pattern, "glob": glob})
        result = self._backend.grep_raw(pattern, path=path, glob=glob)
        if _should_log_path(path):
            trace_log("AgentBackend", "Grep path result", {"path": path, "result_type": type(result).__name__})
        return result

    async def agrep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
        if _should_log_path(path):
            trace_log("AgentBackend", "Grep path", {"path": path, "pattern": pattern, "glob": glob})
        result = await self._backend.agrep_raw(pattern, path=path, glob=glob)
        if _should_log_path(path):
            trace_log("AgentBackend", "Grep path result", {"path": path, "result_type": type(result).__name__})
        return result

    def execute(self, command: str, *, timeout: int | None = None):
        trace_log("AgentBackend", "Executing backend command", {"command": command, "timeout": timeout})
        result = self._backend.execute(command, timeout=timeout)
        trace_log("AgentBackend", "Execute backend command result", {"result_type": type(result).__name__})
        return result

    async def aexecute(self, command: str, *, timeout: int | None = None):
        trace_log("AgentBackend", "Executing backend command", {"command": command, "timeout": timeout})
        result = await self._backend.aexecute(command, timeout=timeout)
        trace_log("AgentBackend", "Execute backend command result", {"result_type": type(result).__name__})
        return result
