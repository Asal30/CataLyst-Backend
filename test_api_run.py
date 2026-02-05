import requests
import os
import json

BASE_URL = "http://127.0.0.1:8001"
IMAGE_PATH = "test_image.jpg"

def test_flow():
    print("=== TEST 1: Workflow (Upload -> Predict -> Explain) ===")
    
    # 1. Upload Image
    if not os.path.exists(IMAGE_PATH):
        from PIL import Image
        Image.new('RGB', (224, 224), color='red').save(IMAGE_PATH)

    files = {'file': open(IMAGE_PATH, 'rb')}
    print(f"\n[1/3] Uploading {IMAGE_PATH}...")
    response = requests.post(f"{BASE_URL}/upload-image", files=files)
    
    if response.status_code != 200:
        print(f"Upload failed: {response.text}")
        return

    data = response.json()
    image_id = data.get("image_id")
    print(f"Upload successful. Image ID: {image_id}")

    # 2. Predict
    print(f"\n[2/3] Requesting prediction for image_id={image_id}...")
    response = requests.post(f"{BASE_URL}/predict", params={"image_id": image_id})

    if response.status_code != 200:
        print(f"Prediction failed: {response.text}")
    else:
        print("Prediction Result:")
        print(json.dumps(response.json(), indent=2))

    # 3. Explain
    print(f"\n[3/3] Requesting explanation for image_id={image_id}...")
    response = requests.post(f"{BASE_URL}/explain", params={"image_id": image_id})

    if response.status_code != 200:
        print(f"Explain failed: {response.text}")
    else:
        print("Explain Result:")
        print(json.dumps(response.json(), indent=2))
        
        gradcam_url = response.json().get("gradcam_url")
        if gradcam_url:
            print(f"Verifying Grad-CAM URL: {BASE_URL}{gradcam_url} ...")
            img_response = requests.get(f"{BASE_URL}{gradcam_url}")
            if img_response.status_code == 200:
                 print("Grad-CAM image is accessible.")
            else:
                 print(f"Could not fetch Grad-CAM image: {img_response.status_code}")

    print("\n\n=== TEST 2: All-in-One (/analyze) ===")
    
    # Analyze Endpoint
    files_analyze = {'file': open(IMAGE_PATH, 'rb')}
    print(f"\n[1/1] Calling /analyze with {IMAGE_PATH}...")
    response = requests.post(f"{BASE_URL}/analyze", files=files_analyze)

    if response.status_code != 200:
        print(f"Analyze failed: {response.text}")
    else:
        print("Analyze Result:")
        print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_flow()
