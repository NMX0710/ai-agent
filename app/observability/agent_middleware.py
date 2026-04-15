from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from app.observability.tracing import is_terminal_trace_enabled, trace_log


def _tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        function_block = tool.get("function")
        if isinstance(function_block, dict) and function_block.get("name"):
            return str(function_block["name"])
        if tool.get("name"):
            return str(tool["name"])
        return "dict_tool"
    return str(getattr(tool, "name", type(tool).__name__))


def _message_text(message: BaseMessage | None) -> str:
    if message is None:
        return ""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    chunks.append(str(text))
            elif isinstance(part, str):
                chunks.append(part)
        return "\n".join(chunks).strip()
    return str(content or "")


def _preview_text(text: str, limit: int = 1600) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...<truncated {len(text) - limit} chars>"


def _normalize_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return {
            "role": message.get("role"),
            "content": message.get("content", ""),
        }
    return {
        "role": getattr(message, "type", None) or getattr(message, "role", None),
        "content": getattr(message, "content", ""),
    }


def _message_preview(message: Any) -> dict[str, Any]:
    normalized = _normalize_message(message)
    return {
        "role": normalized.get("role"),
        "content_preview": _preview_text(_message_text(message)),
    }


def _skills_summary(skills_metadata: Any) -> list[dict[str, Any]]:
    if not isinstance(skills_metadata, list):
        return []

    out: list[dict[str, Any]] = []
    for item in skills_metadata:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "name": item.get("name"),
                "path": item.get("path"),
                "allowed_tools": item.get("allowed_tools") or [],
            }
        )
    return out


def _extract_model_response(response: Any) -> ModelResponse[Any] | None:
    if isinstance(response, ModelResponse):
        return response
    if isinstance(response, ExtendedModelResponse):
        return response.model_response
    if isinstance(response, AIMessage):
        return ModelResponse(result=[response])
    return None


class AgentContextTraceMiddleware(AgentMiddleware):
    def __init__(self, agent_name: str = "agent"):
        super().__init__()
        self.agent_name = agent_name

    def _log_model_request(self, request: ModelRequest[Any]) -> None:
        if not is_terminal_trace_enabled():
            return

        system_text = request.system_message.text if request.system_message else ""
        trace_log(
            "AgentContext",
            "Prepared model request",
            {
                "model": getattr(request.model, "model_name", None)
                or getattr(request.model, "model", None)
                or type(request.model).__name__,
                "agent_name": self.agent_name,
                "message_count": len(request.messages),
                "tool_count": len(request.tools),
                "tools": [_tool_name(tool) for tool in request.tools],
                "state_keys": sorted(request.state.keys()) if isinstance(request.state, dict) else [],
                "skills_metadata_count": len(request.state.get("skills_metadata", []))
                if isinstance(request.state, dict)
                else 0,
                "skills_metadata": _skills_summary(request.state.get("skills_metadata", []))
                if isinstance(request.state, dict)
                else [],
                "system_has_skills_section": "## Skills System" in system_text,
                "system_preview": _preview_text(system_text),
                "last_message_preview": _preview_text(_message_text(request.messages[-1])) if request.messages else "",
            },
        )

    def _log_model_response(self, response: Any) -> None:
        if not is_terminal_trace_enabled():
            return

        model_response = _extract_model_response(response)
        if model_response is None:
            trace_log("AgentContext", "Model response returned unknown shape", {"type": type(response).__name__})
            return

        ai_messages = [msg for msg in model_response.result if isinstance(msg, AIMessage)]
        last_ai = ai_messages[-1] if ai_messages else None
        trace_log(
            "AgentContext",
            "Model response received",
            {
                "agent_name": self.agent_name,
                "result_message_count": len(model_response.result),
                "ai_message_count": len(ai_messages),
                "tool_calls": [
                    {
                        "id": call.get("id"),
                        "name": call.get("name"),
                        "args": call.get("args"),
                    }
                    for call in getattr(last_ai, "tool_calls", [])
                ]
                if last_ai
                else [],
                "assistant_preview": _preview_text(_message_text(last_ai)),
            },
        )

    def _log_tool_request(self, request: ToolCallRequest) -> None:
        if not is_terminal_trace_enabled():
            return

        trace_log(
            "AgentTool",
            "Starting tool execution",
            {
                "agent_name": self.agent_name,
                "tool_name": request.tool_call.get("name"),
                "tool_args": request.tool_call.get("args"),
                "tool_registered": request.tool is not None,
            },
        )

    def _log_tool_response(self, request: ToolCallRequest, result: Any) -> None:
        if not is_terminal_trace_enabled():
            return

        if isinstance(result, ToolMessage):
            payload = {
                "agent_name": self.agent_name,
                "tool_name": request.tool_call.get("name"),
                "status": getattr(result, "status", None),
                "content_preview": _preview_text(_message_text(result)),
            }
        elif isinstance(result, Command):
            update = result.update if isinstance(result.update, dict) else {}
            messages = update.get("messages") if isinstance(update, dict) else None
            payload = {
                "agent_name": self.agent_name,
                "tool_name": request.tool_call.get("name"),
                "result_type": type(result).__name__,
                "command_update_keys": sorted(update.keys()) if isinstance(update, dict) else [],
                "command_messages_count": len(messages) if isinstance(messages, list) else 0,
                "command_last_message": _message_preview(messages[-1]) if isinstance(messages, list) and messages else None,
                "command_goto": str(result.goto) if result.goto else None,
            }
        else:
            payload = {
                "agent_name": self.agent_name,
                "tool_name": request.tool_call.get("name"),
                "result_type": type(result).__name__,
                "result_preview": _preview_text(str(result)),
            }
        trace_log("AgentTool", "Finished tool execution", payload)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler,
    ):
        self._log_model_request(request)
        response = handler(request)
        self._log_model_response(response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler,
    ):
        self._log_model_request(request)
        response = await handler(request)
        self._log_model_response(response)
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ):
        self._log_tool_request(request)
        result = handler(request)
        self._log_tool_response(request, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ):
        self._log_tool_request(request)
        result = await handler(request)
        self._log_tool_response(request, result)
        return result
