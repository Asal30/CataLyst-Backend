from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io
import json

client = TestClient(app)

IMAGE_PATH = "test_image_local.jpg"

# create a test image
img = Image.new('RGB', (224, 224), color='red')
buf = io.BytesIO()
img.save(buf, format='JPEG')
buf.seek(0)

files = {'file': ('test.jpg', buf, 'image/jpeg')}

print('=== TEST 1: Workflow (Upload -> Predict -> Explain) ===')

# Upload
print('\n[1/3] Uploading image...')
resp = client.post('/upload-image', files=files)
print('Upload status:', resp.status_code)
print(resp.json())
if resp.status_code != 200:
    raise SystemExit('Upload failed')

image_id = resp.json().get('image_id')

# Predict
print('\n[2/3] Predicting...')
resp = client.post('/predict', params={'image_id': image_id})
print('Predict status:', resp.status_code)
print(json.dumps(resp.json(), indent=2))

# Explain
print('\n[3/3] Explaining...')
resp = client.post('/explain', params={'image_id': image_id})
print('Explain status:', resp.status_code)
print(json.dumps(resp.json(), indent=2))

# Analyze
print('\n=== TEST 2: /analyze ===')
# need to recreate file buffer
buf.seek(0)
files = {'file': ('test.jpg', buf, 'image/jpeg')}
resp = client.post('/analyze', files=files, data={'source': 'mobile'})
print('Analyze status:', resp.status_code)
print(json.dumps(resp.json(), indent=2))
