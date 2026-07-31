"""
Quick standalone test to confirm your local .env is correctly
talking to your cloud Supabase project.

Run from your project root (where .env lives):
    python test_supabase_connection.py

This does NOT touch app/models/schemas.py or app/config.py.
It only checks: env vars load, client connects, storage upload works,
and a row can be inserted + read back + deleted from `documents` and `memories`.
"""
import os
import tempfile
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ---- 1. Check env vars are present ----
required = ["SUPABASE_URL", "SUPABASE_SECRET_KEY", "SUPABASE_STORAGE_BUCKET"]
missing = [k for k in required if not os.environ.get(k)]
if missing:
    raise SystemExit(f"❌ Missing env vars: {missing}. Check your .env file.")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]
SUPABASE_BUCKET = os.environ["SUPABASE_STORAGE_BUCKET"]

print(f"✅ Env vars loaded")
print(f"   URL:    {SUPABASE_URL}")
print(f"   Bucket: {SUPABASE_BUCKET}")
print(f"   Key:    {SUPABASE_SECRET_KEY[:12]}...(hidden)")

# ---- 2. Connect ----
client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
print("✅ Client created")

# ---- 3. Test storage upload ----
test_filename = f"test_{uuid.uuid4().hex[:8]}.txt"
local_test_path = os.path.join(tempfile.gettempdir(), "test_upload.txt")
with open(local_test_path, "w") as f:
    f.write("hello from local test script")

try:
    client.storage.from_(SUPABASE_BUCKET).upload(test_filename, open(local_test_path, "rb"))
    print(f"✅ Uploaded test file to storage: {test_filename}")
except Exception as e:
    print(f"❌ Storage upload failed: {e}")
    raise SystemExit(1)

# Clean up the test file from storage
client.storage.from_(SUPABASE_BUCKET).remove([test_filename])
print("✅ Cleaned up test file from storage")

# ---- 4. Test documents table insert/select/delete ----
doc_id = f"test-doc-{uuid.uuid4().hex[:8]}"
doc_row = {
    "doc_id": doc_id,
    "filename": "test.txt",
    "file_type": "text/plain",
    "user_id": "test-user",
    "raw_storage_path": test_filename,
    "uploaded_at": datetime.now(timezone.utc).isoformat(),
}

try:
    client.table("documents").insert(doc_row).execute()
    print(f"✅ Inserted row into documents: {doc_id}")

    res = client.table("documents").select("*").eq("doc_id", doc_id).execute()
    assert res.data and res.data[0]["doc_id"] == doc_id
    print("✅ Read row back from documents")

    client.table("documents").delete().eq("doc_id", doc_id).execute()
    print("✅ Deleted test row from documents")
except Exception as e:
    print(f"❌ documents table test failed: {e}")
    raise SystemExit(1)

# ---- 5. Test memories table insert/select/delete ----
memory_id = f"test-mem-{uuid.uuid4().hex[:8]}"
memory_row = {
    "memory_id": memory_id,
    "doc_id": None,
    "text": "this is a test memory",
    "status": "active",
    "superseded_by": None,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}

try:
    client.table("memories").insert(memory_row).execute()
    print(f"✅ Inserted row into memories: {memory_id}")

    res = client.table("memories").select("*").eq("memory_id", memory_id).execute()
    assert res.data and res.data[0]["memory_id"] == memory_id
    print("✅ Read row back from memories")

    client.table("memories").delete().eq("memory_id", memory_id).execute()
    print("✅ Deleted test row from memories")
except Exception as e:
    print(f"❌ memories table test failed: {e}")
    raise SystemExit(1)

print("\n🎉 All checks passed — local .env is correctly wired to your cloud Supabase project.")