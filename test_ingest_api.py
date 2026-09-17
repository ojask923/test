import httpx
import json

base_url = "http://127.0.0.1:8000/api"

print("--- 1. Ingest normal ---")
with open("test_ingest.txt", "rb") as f:
    files = {"file": ("test_ingest.txt", f, "text/plain")}
    data = {"user_id": "test_user_1", "session_id": "test_sess_1"}
    r = httpx.post(f"{base_url}/rag/ingest", files=files, data=data)
    print(r.status_code)
    print(json.dumps(r.json(), indent=2))
    doc_id = r.json().get("document_id")

print("\n--- 2. Ingest duplicate ---")
with open("test_ingest.txt", "rb") as f:
    files = {"file": ("test_ingest.txt", f, "text/plain")}
    data = {"user_id": "test_user_1", "session_id": "test_sess_1"}
    r = httpx.post(f"{base_url}/rag/ingest", files=files, data=data)
    print(r.status_code)
    print(json.dumps(r.json(), indent=2))

print("\n--- 3. Ingest force ---")
with open("test_ingest.txt", "rb") as f:
    files = {"file": ("test_ingest.txt", f, "text/plain")}
    data = {"user_id": "test_user_1", "session_id": "test_sess_1"}
    r = httpx.post(f"{base_url}/rag/ingest?force=true", files=files, data=data)
    print(r.status_code)
    print(json.dumps(r.json(), indent=2))
    doc_id = r.json().get("document_id")

print("\n--- 4. List docs ---")
r = httpx.get(f"{base_url}/rag/documents")
print(r.status_code)
print(json.dumps(r.json(), indent=2))

print(f"\n--- 5. Get single doc {doc_id} ---")
r = httpx.get(f"{base_url}/rag/documents/{doc_id}")
print(r.status_code)
print(json.dumps(r.json(), indent=2))

print("\n--- 6. Delete doc ---")
r = httpx.delete(f"{base_url}/rag/documents/{doc_id}")
print(r.status_code)
print(json.dumps(r.json(), indent=2))

print("\n--- 7. Verify deletion ---")
r = httpx.get(f"{base_url}/rag/documents/{doc_id}")
print(r.status_code)
print(json.dumps(r.json(), indent=2))
