"""
Quick standalone test to confirm your local .env is correctly
talking to your FalkorDB Cloud instance.

Run from your project root (where .env lives):
    python test_falkordb_connection.py

Requires: pip install FalkorDB python-dotenv
"""
import os
import uuid

from dotenv import load_dotenv
from falkordb import FalkorDB

load_dotenv()

# ---- 1. Check env vars ----
required = ["FALKORDB_HOST", "FALKORDB_PORT", "FALKORDB_PASSWORD"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    raise SystemExit(f"❌ Missing env vars: {missing}. Check your .env file.")

HOST = os.environ["FALKORDB_HOST"]
PORT = int(os.environ["FALKORDB_PORT"])
USERNAME = os.environ.get("FALKORDB_USERNAME", "falkordb")
PASSWORD = os.environ["FALKORDB_PASSWORD"]

print("✅ Env vars loaded")
print(f"   Host: {HOST}")
print(f"   Port: {PORT}")
print(f"   User: {USERNAME}")
print(f"   Pass: {'*' * len(PASSWORD)}")

# ---- 2. Connect ----
try:
    db = FalkorDB(host=HOST, port=PORT, username=USERNAME, password=PASSWORD)
    print("✅ Client created")
except Exception as e:
    print(f"❌ Failed to create client: {e}")
    raise SystemExit(1)

# ---- 3. Select/create a test graph ----
graph_name = f"test_graph_{uuid.uuid4().hex[:6]}"
try:
    g = db.select_graph(graph_name)
    print(f"✅ Selected graph: {graph_name}")
except Exception as e:
    print(f"❌ Failed to select graph: {e}")
    raise SystemExit(1)

# ---- 4. Create test nodes + relationship ----
try:
    g.query("""
        CREATE (a:TestEntity {name: 'Alice', id: 1})
        CREATE (b:TestEntity {name: 'Bob', id: 2})
        CREATE (a)-[:KNOWS {since: 2024}]->(b)
    """)
    print("✅ Created test nodes and relationship")
except Exception as e:
    print(f"❌ Failed to write test data: {e}")
    raise SystemExit(1)

# ---- 5. Query it back ----
try:
    result = g.query("""
        MATCH (a:TestEntity)-[r:KNOWS]->(b:TestEntity)
        RETURN a.name, b.name, r.since
    """)
    rows = result.result_set
    assert len(rows) == 1
    name_a, name_b, since = rows[0]
    print(f"✅ Query returned: {name_a} KNOWS {name_b} since {since}")
except Exception as e:
    print(f"❌ Failed to query test data: {e}")
    raise SystemExit(1)

# ---- 6. Two-hop style traversal check (sanity check for your graph expansion feature) ----
try:
    g.query("""
        MATCH (b:TestEntity {name: 'Bob'})
        CREATE (c:TestEntity {name: 'Charlie', id: 3})
        CREATE (b)-[:KNOWS {since: 2025}]->(c)
    """)
    result = g.query("""
        MATCH (a:TestEntity {name: 'Alice'})-[:KNOWS*1..2]->(x:TestEntity)
        RETURN x.name
    """)
    names = sorted(row[0] for row in result.result_set)
    assert names == ["Bob", "Charlie"]
    print(f"✅ Two-hop traversal works: Alice -> {names}")
except Exception as e:
    print(f"❌ Two-hop traversal failed: {e}")
    raise SystemExit(1)

# ---- 7. Clean up ----
try:
    g.delete()
    print(f"✅ Deleted test graph: {graph_name}")
except Exception as e:
    print(f"⚠️  Could not delete test graph (not fatal): {e}")

print("\n🎉 All checks passed — local .env is correctly wired to your FalkorDB Cloud instance.")