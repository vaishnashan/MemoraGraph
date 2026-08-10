"""
FalkorDB client wrapper. FalkorDB speaks OpenCypher, so all queries below
are Cypher — useful to know if you want to poke around with FalkorDB's
own browser UI too.
"""
from falkordb import FalkorDB
from redis.exceptions import ConnectionError as RedisConnectionError

from application.ingestion.config import settings
from application.ingestion.schemas import Entity, Relationship

_graph_cache = None


def get_graph():
    """
    Cached so we don't reconnect to FalkorDB on every single entity/relationship
    write — one connection is reused for the life of the process.

    Prints exactly what host/port/ssl it's about to connect with, and raises a
    clear message (instead of a deep traceback) if the connection fails —
    almost always a wrong/stale FALKORDB_HOST value or a paused/deleted
    instance, not a bug in this code.
    """
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache

    print(
        f"[falkordb] connecting to host={settings.falkordb_host!r} "
        f"port={settings.falkordb_port} ssl={settings.falkordb_ssl}"
    )
    try:
        db = FalkorDB(
            host=settings.falkordb_host,
            port=settings.falkordb_port,
            username=settings.falkordb_username,
            password=settings.falkordb_password,
            ssl=settings.falkordb_ssl,
            socket_connect_timeout=6,   # fail fast instead of hanging on a blocked port
            socket_timeout=6,
        )
        _graph_cache = db.select_graph(settings.falkordb_graph_name)
        print("[falkordb] connected OK")
        return _graph_cache
    except RedisConnectionError as e:
        raise RuntimeError(
            f"[falkordb] Could not connect to FALKORDB_HOST={settings.falkordb_host!r} "
            f"on port {settings.falkordb_port}. This is almost always a wrong/stale "
            f"host in .env, or the FalkorDB instance being paused/deleted — check "
            f"your FalkorDB dashboard and re-copy the host. Original error: {e}"
        ) from e


def add_entity(entity: Entity) -> str:
    """
    Adds or resolves an entity node.

    Entity resolution / duplicate detection: merges on (lowercased name,
    type) rather than the freshly-generated entity_id, so the same
    real-world entity mentioned across many chunks/documents becomes ONE
    graph node instead of a new duplicate node every time it's seen.

    Returns the CANONICAL entity_id — either the existing one if this
    entity was already known, or the newly created one. Callers must use
    this returned id (not entity.entity_id) when building relationships,
    since the id may have been remapped to an existing node.
    """
    graph = get_graph()
    result = graph.query(
        """
        MERGE (e:Entity {name_key: toLower($name), type: $type})
        ON CREATE SET
            e.entity_id = $entity_id,
            e.name = $name,
            e.source_doc_id = $source_doc_id
        RETURN e.entity_id
        """,
        params={
            "entity_id": entity.entity_id,
            "name": entity.name,
            "type": entity.type,
            "source_doc_id": entity.source_doc_id,
        },
    )
    return result.result_set[0][0]


def add_relationship(rel: Relationship) -> None:
    """
    IMPORTANT: source_entity_id / target_entity_id must be the CANONICAL
    ids returned by add_entity() — not the raw ids from extraction — or
    this MATCH will silently find nothing and no relationship gets created.
    """
    graph = get_graph()
    graph.query(
        """
        MATCH (a:Entity {entity_id: $source_id})
        MATCH (b:Entity {entity_id: $target_id})
        MERGE (a)-[r:RELATES {type: $relation}]->(b)
        SET r.source_doc_id = $source_doc_id
        """,
        params={
            "source_id": rel.source_entity_id,
            "target_id": rel.target_entity_id,
            "relation": rel.relation,
            "source_doc_id": rel.source_doc_id,
        },
    )


def find_entity_by_name(name: str) -> dict | None:
    graph = get_graph()
    result = graph.query(
        "MATCH (e:Entity) WHERE toLower(e.name) = toLower($name) "
        "RETURN e.entity_id, e.name, e.type LIMIT 1",
        params={"name": name},
    )
    if not result.result_set:
        return None
    row = result.result_set[0]
    return {"entity_id": row[0], "name": row[1], "type": row[2]}


def get_entities_by_doc_id(doc_id: str) -> list[dict]:
    """All entities extracted from a given source document — used by
    get_document_context to show what MemoraGraph learned from a file."""
    graph = get_graph()
    result = graph.query(
        "MATCH (e:Entity {source_doc_id: $doc_id}) RETURN e.entity_id, e.name, e.type",
        params={"doc_id": doc_id},
    )
    return [{"entity_id": row[0], "name": row[1], "type": row[2]} for row in result.result_set]


def two_hop_expansion(entity_name: str) -> list[dict]:
    """
    Pulls up to two relationship hops out from a matched entity.
    Example from the project spec:
      NOVA -> uses -> LangGraph
      NOVA -> deployed_on -> GCP
    """
    graph = get_graph()
    result = graph.query(
        """
        MATCH (start:Entity {name: $name})-[r1:RELATES]->(hop1)
        OPTIONAL MATCH (hop1)-[r2:RELATES]->(hop2)
        RETURN start.name, r1.type, hop1.name, hop1.type, r2.type, hop2.name, hop2.type
        """,
        params={"name": entity_name},
    )

    related = []
    for row in result.result_set:
        start_name, rel1, hop1_name, hop1_type, rel2, hop2_name, hop2_type = row
        related.append({
            "name": hop1_name,
            "type": hop1_type,
            "relation_path": [rel1],
        })
        if hop2_name:
            related.append({
                "name": hop2_name,
                "type": hop2_type,
                "relation_path": [rel1, rel2],
            })
    return related
