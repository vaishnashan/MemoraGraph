"""
User-level access control + audit logging for MCP tool calls.

Kept simple to start: every write/delete call goes through
check_permission() first, and every call (read or write) gets logged
through audit_log() so you can show "which MCP tool accessed which
memory" per the spec.

NOTE: audit log is in-memory only right now — it resets whenever the MCP
server process restarts. That's fine for local testing. Before deploying,
swap _AUDIT_LOG for a Supabase table (see the `audit_logs` note in
schema.sql) so it persists and survives restarts.
"""
import datetime
from enum import Enum


class ToolRisk(str, Enum):
    READ = "read"          # search_memory_hybrid, search_memory_semantic, find_related_entities, get_document_context
    WRITE = "write"         # store_memory
    SENSITIVE = "sensitive"  # update_memory, forget_memory


_AUDIT_LOG: list[dict] = []


class PermissionDenied(Exception):
    pass


def check_permission(user_id: str, tool_name: str, risk: ToolRisk) -> None:
    """
    Placeholder access control. Every known user_id can do reads and writes;
    sensitive ops (update/forget) require the user to be the owner of the
    memory being touched — enforce that at the call site once you have real
    user/memory ownership data.
    """
    if not user_id:
        raise PermissionDenied(f"No user_id provided for tool '{tool_name}'.")
    # Extend here: role checks, per-memory ownership checks, rate limits, etc.


def audit_log(user_id: str, tool_name: str, memory_id: str | None, action: str) -> None:
    entry = {
        "user_id": user_id,
        "tool_name": tool_name,
        "memory_id": memory_id,
        "action": action,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    _AUDIT_LOG.append(entry)


def get_audit_log() -> list[dict]:
    return _AUDIT_LOG
