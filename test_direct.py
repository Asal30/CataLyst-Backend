from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("Testing /health...")
response = client.get("/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")

print("\nTesting /predict...")
response = client.post("/predict?image_id=test123&source=mobile")
print(f"Status: {response.status_code}")
print(f"Response: {response.json() if response.status_code == 200 else response.text}")