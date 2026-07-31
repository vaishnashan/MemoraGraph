"""
Extracts entities and relationships from a text chunk using an LLM,
at write time (upload), not at query time — keeps retrieval fast.

Provider: Google Gemini (Google AI Studio free tier, gemini-2.5-flash by default)

Required env vars:
    GEMINI_API_KEY
    EXTRACTION_MODEL   (optional override, defaults to gemini-2.5-flash)
"""
import json
import os
import uuid

from dotenv import load_dotenv
from google import genai

from application.models6.schemas import Entity, Relationship

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "gemini-2.5-flash")

EXTRACTION_PROMPT = """Extract entities and relationships from the text below.
Return ONLY valid JSON, no other text, in this exact shape:

{{
  "entities": [{{"name": "...", "type": "person|project|concept|deadline|organization|task"}}],
  "relationships": [{{"source": "...", "target": "...", "relation": "..."}}]
}}

Text:
{text}
"""


def _call_gemini(prompt: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=EXTRACTION_MODEL,
        contents=prompt,
    )
    return response.text


def extract_entities_and_relationships(
    text: str, doc_id: str
) -> tuple[list[Entity], list[Relationship]]:
    prompt = EXTRACTION_PROMPT.format(text=text)
    raw = _call_gemini(prompt)

    # Strip potential markdown fences before parsing
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Extraction models occasionally wrap or malform JSON — fail soft
        return [], []

    name_to_id: dict[str, str] = {}
    entities: list[Entity] = []
    for e in parsed.get("entities", []):
        entity_id = str(uuid.uuid4())
        name_to_id[e["name"]] = entity_id
        entities.append(Entity(
            entity_id=entity_id,
            name=e["name"],
            type=e.get("type", "concept"),
            source_doc_id=doc_id,
        ))

    relationships: list[Relationship] = []
    for r in parsed.get("relationships", []):
        source_id = name_to_id.get(r["source"])
        target_id = name_to_id.get(r["target"])
        if source_id and target_id:
            relationships.append(Relationship(
                source_entity_id=source_id,
                target_entity_id=target_id,
                relation=r["relation"],
                source_doc_id=doc_id,
            ))

    return entities, relationships