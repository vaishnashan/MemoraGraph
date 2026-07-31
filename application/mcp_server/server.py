"""
MCP server — the "doorway" that lets external AI agents (Claude, a
LangGraph app, etc.) call your backend as tools, without knowing how
any of it is implemented internally.

Exposes 6 tools:
  1. store_memory          (write)
  2. search_memory          (read)
  3. find_related_entities  (read)
  4. get_document_context   (read)
  5. update_memory          (sensitive)
  6. forget_memory          (sensitive, requires confirm=true)

Run with:
    python -m application.mcp_server.server
"""
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json

from application.memory.lifecycle import store_memory, update_memory, forget_memory, get_document_context
from application.retrieval.rrf_fusion import hybrid_search
from application.retrieval.vector_search import vector_index
from application.retrieval.bm25_search import bm25_index
from application.retrieval.fuzzy_search import fuzzy_index
from application.graph2.falkordb_client import two_hop_expansion
from application.security.permissions import check_permission, audit_log, ToolRisk, PermissionDenied

server = Server("memoragraph")

TOOLS = [
    Tool(
        name="store_memory",
        description="Store a new memory (fact, note, or extracted info) in MemoraGraph.",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "text": {"type": "string"},
                "source_doc_id": {"type": "string"},
            },
            "required": ["user_id", "text"],
        },
    ),
    Tool(
        name="search_memory",
        description="Search stored memories using hybrid retrieval (vector + BM25 + fuzzy) and return ranked results with citations.",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["user_id", "query"],
        },
    ),
    Tool(
        name="find_related_entities",
        description="Given an entity name, return related entities up to two relationship hops away in the knowledge graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "entity_name": {"type": "string"},
            },
            "required": ["entity_name"],
        },
    ),
    Tool(
        name="get_document_context",
        description="Given a document id, return everything MemoraGraph knows about it: metadata, extracted entities, and stored chunk content.",
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
            },
            "required": ["doc_id"],
        },
    ),
    Tool(
        name="update_memory",
        description="Update an existing memory. Marks the old version as superseded and stores the new text as an active memory.",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "memory_id": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["user_id", "memory_id", "new_text"],
        },
    ),
    Tool(
        name="forget_memory",
        description="Delete a memory permanently. Requires confirm=true — the caller must confirm the deletion with the user first.",
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string"},
                "memory_id": {"type": "string"},
                "confirm": {"type": "boolean", "default": False},
            },
            "required": ["user_id", "memory_id"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    user_id = arguments.get("user_id", "")

    try:
        if name == "store_memory":
            check_permission(user_id, name, ToolRisk.WRITE)
            record, duplicate_id = store_memory(arguments["text"], arguments.get("source_doc_id"))
            audit_log(user_id, name, record.memory_id, "store")
            result = {
                "memory_id": record.memory_id,
                "status": record.status,
                "duplicate_of": duplicate_id,
            }

        elif name == "search_memory":
            check_permission(user_id, name, ToolRisk.READ)
            top_k = arguments.get("top_k", 5)
            fused = hybrid_search(arguments["query"], vector_index, bm25_index, fuzzy_index, top_k=top_k)
            results = [
                {"memory_id": cid, "text": vector_index.get_text(cid), "score": round(score, 4)}
                for cid, score in fused
            ]
            audit_log(user_id, name, None, f"search: {arguments['query']}")
            result = {"results": results}

        elif name == "find_related_entities":
            related = two_hop_expansion(arguments["entity_name"])
            audit_log(user_id or "anonymous", name, None, f"related: {arguments['entity_name']}")
            result = {"related": related}

        elif name == "get_document_context":
            check_permission(user_id or "anonymous", name, ToolRisk.READ)
            context = get_document_context(arguments["doc_id"])
            audit_log(user_id or "anonymous", name, arguments["doc_id"], "get_document_context")
            result = context

        elif name == "update_memory":
            check_permission(user_id, name, ToolRisk.SENSITIVE)
            new_record = update_memory(arguments["memory_id"], arguments["new_text"])
            audit_log(user_id, name, arguments["memory_id"], "update")
            result = {
                "old_memory_id": arguments["memory_id"],
                "new_memory_id": new_record.memory_id,
                "status": new_record.status,
            }

        elif name == "forget_memory":
            check_permission(user_id, name, ToolRisk.SENSITIVE)
            confirm = arguments.get("confirm", False)
            deleted, message = forget_memory(arguments["memory_id"], confirmed=confirm)
            audit_log(user_id, name, arguments["memory_id"], f"forget (confirmed={confirm})")
            result = {"memory_id": arguments["memory_id"], "deleted": deleted, "message": message}

        else:
            result = {"error": f"Unknown tool: {name}"}

    except PermissionDenied as e:
        result = {"error": str(e)}

    return [TextContent(type="text", text=json.dumps(result))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())