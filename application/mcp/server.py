"""
MemoraGraph MCP Server

This server exposes MemoraGraph operations as MCP tools so external
AI agents such as Claude or LangGraph applications can interact with
the memory system.

Exposes 11 tools:

1.  store_memory
2.  search_memory_hybrid
3.  search_memory_semantic
4.  find_related_entities
5.  find_entity
6.  get_document_context
7.  list_documents
8.  list_memories
9.  update_memory
10. forget_memory
11. get_audit_log

Run:

    python -m application.mcp.server
"""

import asyncio
import json
from typing import Any

import mcp.server.stdio

from mcp.server.lowlevel import (
    NotificationOptions,
    Server,
)

from mcp.server.models import InitializationOptions

from mcp.types import (
    Tool,
    TextContent,
)


# ============================================================
# APPLICATION IMPORTS
# ============================================================

from application.mcp.lifecycle import (
    store_memory,
    update_memory,
    forget_memory,
    get_document_context,
)

from application.mcp.rrf_fusion import hybrid_search

from application.mcp.vector_search import vector_index
from application.mcp.bm25_search import bm25_index
from application.mcp.fuzzy_search import fuzzy_index

from application.mcp.falkordb_client import (
    two_hop_expansion,
    find_entity_by_name,
)

from application.mcp import supabase_client as db

from application.mcp.security import (
    check_permission,
    audit_log,
    get_audit_log,
    ToolRisk,
    PermissionDenied,
)


# ============================================================
# CREATE MCP SERVER
# ============================================================

server = Server("memoragraph")


# ============================================================
# MCP TOOL DEFINITIONS
# ============================================================

TOOLS = [

    # ========================================================
    # 1. STORE MEMORY
    # ========================================================

    Tool(
        name="store_memory",
        description=(
            "Store a new memory (fact, note, or extracted info) "
            "in MemoraGraph."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                },
                "text": {
                    "type": "string",
                },
                "source_doc_id": {
                    "type": "string",
                },
            },
            "required": [
                "user_id",
                "text",
            ],
        },
    ),


    # ========================================================
    # 2. SEARCH MEMORY (HYBRID)
    # ========================================================

    Tool(
        name="search_memory_hybrid",
        description=(
            "Search stored memories using fused hybrid retrieval "
            "(vector + BM25 + fuzzy, combined with RRF) and return "
            "ranked results. Best general-purpose search."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                },
                "query": {
                    "type": "string",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                },
            },
            "required": [
                "user_id",
                "query",
            ],
        },
    ),


    # ========================================================
    # 3. SEARCH MEMORY (SEMANTIC)
    # ========================================================

    Tool(
        name="search_memory_semantic",
        description=(
            "Perform raw semantic vector search over stored "
            "chunks/memories without BM25 or fuzzy fusion."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                },
                "query": {
                    "type": "string",
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                },
            },
            "required": [
                "user_id",
                "query",
            ],
        },
    ),


    # ========================================================
    # 4. FIND RELATED ENTITIES
    # ========================================================

    Tool(
        name="find_related_entities",
        description=(
            "Given an entity name, return related entities up to "
            "two relationship hops away in the knowledge graph."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                },
            },
            "required": [
                "entity_name",
            ],
        },
    ),


    # ========================================================
    # 5. FIND ENTITY
    # ========================================================

    Tool(
        name="find_entity",
        description=(
            "Look up a single entity directly by name without "
            "relationship expansion."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                },
            },
            "required": [
                "entity_name",
            ],
        },
    ),


    # ========================================================
    # 6. GET DOCUMENT CONTEXT
    # ========================================================

    Tool(
        name="get_document_context",
        description=(
            "Given a document ID, return everything MemoraGraph "
            "knows about the document including metadata, extracted "
            "entities and stored chunk content."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                },
            },
            "required": [
                "doc_id",
            ],
        },
    ),


    # ========================================================
    # 7. LIST DOCUMENTS
    # ========================================================

    Tool(
        name="list_documents",
        description=(
            "List every document uploaded by a given user with "
            "metadata such as filename, file type and upload time."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                },
            },
            "required": [
                "user_id",
            ],
        },
    ),


    # ========================================================
    # 8. LIST MEMORIES
    # ========================================================

    Tool(
        name="list_memories",
        description=(
            "List all currently active memories for a user."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                },
            },
            "required": [
                "user_id",
            ],
        },
    ),


    # ========================================================
    # 9. UPDATE MEMORY
    # ========================================================

    Tool(
        name="update_memory",
        description=(
            "Update an existing memory. The old memory is marked "
            "as superseded and the new text becomes active."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                },
                "memory_id": {
                    "type": "string",
                },
                "new_text": {
                    "type": "string",
                },
            },
            "required": [
                "user_id",
                "memory_id",
                "new_text",
            ],
        },
    ),


    # ========================================================
    # 10. FORGET MEMORY
    # ========================================================

    Tool(
        name="forget_memory",
        description=(
            "Delete a memory permanently. Requires confirm=true."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                },
                "memory_id": {
                    "type": "string",
                },
                "confirm": {
                    "type": "boolean",
                    "default": False,
                },
            },
            "required": [
                "user_id",
                "memory_id",
            ],
        },
    ),


    # ========================================================
    # 11. GET AUDIT LOG
    # ========================================================

    Tool(
        name="get_audit_log",
        description=(
            "Return the audit trail showing which MCP tools "
            "accessed or modified information."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
]


# ============================================================
# LIST TOOLS
# ============================================================
#
# MCP 1.x uses decorator-based registration.
#
# The MCP SDK automatically converts this list into the proper
# ListToolsResult response.
#
# ============================================================

@server.list_tools()
async def handle_list_tools() -> list[Tool]:

    return TOOLS


# ============================================================
# CALL TOOL
# ============================================================

@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict[str, Any],
) -> list[TextContent]:

    user_id = arguments.get(
        "user_id",
        "",
    )

    try:

        # ====================================================
        # STORE MEMORY
        # ====================================================

        if name == "store_memory":

            check_permission(
                user_id,
                name,
                ToolRisk.WRITE,
            )

            record, duplicate_id = store_memory(
                arguments["text"],
                arguments.get("source_doc_id"),
            )

            audit_log(
                user_id,
                name,
                record.memory_id,
                "store",
            )

            result = {
                "memory_id": record.memory_id,
                "status": record.status,
                "duplicate_of": duplicate_id,
            }


        # ====================================================
        # SEARCH MEMORY (HYBRID)
        # Hybrid Retrieval:
        # Vector + BM25 + Fuzzy + RRF
        # ====================================================

        elif name == "search_memory_hybrid":

            check_permission(
                user_id,
                name,
                ToolRisk.READ,
            )

            top_k = arguments.get(
                "top_k",
                5,
            )

            fused = hybrid_search(
                arguments["query"],
                vector_index,
                bm25_index,
                fuzzy_index,
                top_k=top_k,
            )

            results = [
                {
                    "memory_id": memory_id,
                    "text": vector_index.get_text(
                        memory_id
                    ),
                    "score": round(
                        score,
                        4,
                    ),
                }
                for memory_id, score in fused
            ]

            audit_log(
                user_id,
                name,
                None,
                f"search: {arguments['query']}",
            )

            result = {
                "results": results,
            }


        # ====================================================
        # SEARCH MEMORY (SEMANTIC)
        # Vector-only semantic retrieval
        # ====================================================

        elif name == "search_memory_semantic":

            check_permission(
                user_id,
                name,
                ToolRisk.READ,
            )

            top_k = arguments.get(
                "top_k",
                10,
            )

            raw_results = vector_index.search(
                arguments["query"],
                top_k=top_k,
            )

            results = [
                {
                    "item_id": item_id,
                    "text": vector_index.get_text(
                        item_id
                    ),
                    "score": round(
                        score,
                        4,
                    ),
                }
                for item_id, score in raw_results
            ]

            audit_log(
                user_id,
                name,
                None,
                f"search_memory_semantic: {arguments['query']}",
            )

            result = {
                "results": results,
            }


        # ====================================================
        # FIND RELATED ENTITIES
        # FalkorDB two-hop graph expansion
        # ====================================================

        elif name == "find_related_entities":

            related = two_hop_expansion(
                arguments["entity_name"]
            )

            audit_log(
                user_id or "anonymous",
                name,
                None,
                (
                    "related: "
                    f"{arguments['entity_name']}"
                ),
            )

            result = {
                "related": related,
            }


        # ====================================================
        # FIND ENTITY
        # ====================================================

        elif name == "find_entity":

            entity = find_entity_by_name(
                arguments["entity_name"]
            )

            audit_log(
                user_id or "anonymous",
                name,
                None,
                (
                    "find_entity: "
                    f"{arguments['entity_name']}"
                ),
            )

            result = {
                "entity": entity,
            }


        # ====================================================
        # GET DOCUMENT CONTEXT
        # ====================================================

        elif name == "get_document_context":

            check_permission(
                user_id or "anonymous",
                name,
                ToolRisk.READ,
            )

            context = get_document_context(
                arguments["doc_id"]
            )

            audit_log(
                user_id or "anonymous",
                name,
                arguments["doc_id"],
                "get_document_context",
            )

            result = context


        # ====================================================
        # LIST DOCUMENTS
        # ====================================================

        elif name == "list_documents":

            check_permission(
                user_id,
                name,
                ToolRisk.READ,
            )

            documents = (
                db.list_documents_by_user(
                    user_id
                )
            )

            audit_log(
                user_id,
                name,
                None,
                "list_documents",
            )

            result = {
                "documents": [
                    document.model_dump(
                        mode="json"
                    )
                    for document in documents
                ],
            }


        # ====================================================
        # LIST MEMORIES
        # ====================================================

        elif name == "list_memories":

            check_permission(
                user_id,
                name,
                ToolRisk.READ,
            )

            memories = (
                db.list_active_memories(
                    user_id
                )
            )

            audit_log(
                user_id,
                name,
                None,
                "list_memories",
            )

            result = {
                "memories": [
                    memory.model_dump(
                        mode="json"
                    )
                    for memory in memories
                ],
            }


        # ====================================================
        # UPDATE MEMORY
        # ====================================================

        elif name == "update_memory":

            check_permission(
                user_id,
                name,
                ToolRisk.SENSITIVE,
            )

            new_record = update_memory(
                arguments["memory_id"],
                arguments["new_text"],
            )

            audit_log(
                user_id,
                name,
                arguments["memory_id"],
                "update",
            )

            result = {
                "old_memory_id":
                    arguments["memory_id"],

                "new_memory_id":
                    new_record.memory_id,

                "status":
                    new_record.status,
            }


        # ====================================================
        # FORGET MEMORY
        # ====================================================

        elif name == "forget_memory":

            check_permission(
                user_id,
                name,
                ToolRisk.SENSITIVE,
            )

            confirm = arguments.get(
                "confirm",
                False,
            )

            deleted, message = forget_memory(
                arguments["memory_id"],
                confirmed=confirm,
            )

            audit_log(
                user_id,
                name,
                arguments["memory_id"],
                (
                    "forget "
                    f"(confirmed={confirm})"
                ),
            )

            result = {
                "memory_id":
                    arguments["memory_id"],

                "deleted":
                    deleted,

                "message":
                    message,
            }


        # ====================================================
        # GET AUDIT LOG
        # ====================================================

        elif name == "get_audit_log":

            result = {
                "audit_log":
                    get_audit_log(),
            }


        # ====================================================
        # UNKNOWN TOOL
        # ====================================================

        else:

            result = {
                "error":
                    f"Unknown tool: {name}",
            }


    # ========================================================
    # PERMISSION ERROR
    # ========================================================

    except PermissionDenied as error:

        result = {
            "error": str(error),
        }


    # ========================================================
    # RETURN MCP CONTENT
    # ========================================================

    return [
        TextContent(
            type="text",
            text=json.dumps(
                result,
                default=str,
            ),
        )
    ]


# ============================================================
# START MCP SERVER
# ============================================================

async def main():

    async with (
        mcp.server.stdio.stdio_server()
    ) as (
        read_stream,
        write_stream,
    ):

        await server.run(
            read_stream,
            write_stream,

            InitializationOptions(
                server_name="memoragraph",
                server_version="1.0.0",

                capabilities=(
                    server.get_capabilities(
                        notification_options=(
                            NotificationOptions()
                        ),
                        experimental_capabilities={},
                    )
                ),
            ),
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())