"""
FalkorDB client wrapper. FalkorDB speaks OpenCypher, so all queries below
are Cypher — useful to know if you want to poke around with FalkorDB's
own browser UI too.
"""
from falkordb import FalkorDB
from app.config import settings
from app.models.schemas import Entity, Relationship


def get_graph():
    db = FalkorDB(host=settings.falkordb_host, port=settings.falkordb_port)
    return db.select_graph(settings.falkordb_graph_name)


def add_entity(entity: Entity) -> None:
    graph = get_graph()
    graph.query(
        """
        MERGE (e:Entity {entity_id: $entity_id})
        SET e.name = $name, e.type = $type, e.source_doc_id = $source_doc_id
        """,
        params={
            "entity_id": entity.entity_id,
            "name": entity.name,
            "type": entity.type,
            "source_doc_id": entity.source_doc_id,
        },
    )


def add_relationship(rel: Relationship) -> None:
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
        "MATCH (e:Entity {name: $name}) RETURN e.entity_id, e.name, e.type LIMIT 1",
        params={"name": name},
    )
    if not result.result_set:
        return None
    row = result.result_set[0]
    return {"entity_id": row[0], "name": row[1], "type": row[2]}


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
