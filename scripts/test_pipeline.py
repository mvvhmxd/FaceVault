"""Quick test of the full FaceVault pipeline against a sample image."""
import os
import requests
import json

BASE = "http://localhost:8080/api"
LFW = "data/lfw_sample"


def find_test_image():
    if not os.path.exists(LFW):
        return None
    for root, _, files in os.walk(LFW):
        for f in files:
            if f.lower().endswith(".jpg"):
                return os.path.join(root, f)
    return None


def main():
    img_path = find_test_image()
    if not img_path:
        print("No test image found in", LFW)
        return

    print(f"Testing with: {img_path}\n")

    with open(img_path, "rb") as fh:
        files = {"file": ("test.jpg", fh, "image/jpeg")}
        r = requests.post(f"{BASE}/process-frame", files=files)

    print(f"Status: {r.status_code}")
    data = r.json()
    print(f"Faces detected: {data.get('count', 0)}")

    for i, face in enumerate(data.get("faces", [])):
        print(f"\n--- Face #{i+1} ---")
        print(f"  Name:       {face.get('name')}")
        print(f"  Confidence: {face.get('recognition_confidence')}")
        print(f"  Age:        {face.get('age')}")
        print(f"  Gender:     {face.get('gender')}")
        print(f"  Emotion:    {face.get('emotion')} ({face.get('emotion_confidence')})")
        print(f"  Liveness:   {face.get('is_live')} ({face.get('liveness_confidence')})")
        celeb = face.get("celebrity_match")
        if celeb:
            print(f"  Celebrity:  {celeb.get('name')} ({celeb.get('similarity'):.2%})")

    # Test registration
    print("\n--- Registering face as 'TestUser' ---")
    with open(img_path, "rb") as fh:
        r = requests.post(f"{BASE}/register",
                          data={"name": "TestUser"},
                          files={"file": ("reg.jpg", fh, "image/jpeg")})
    print(f"Register: {r.json()}")

    # Test recognition after registration
    print("\n--- Re-processing (should now recognize) ---")
    with open(img_path, "rb") as fh:
        r = requests.post(f"{BASE}/process-frame",
                          files={"file": ("test.jpg", fh, "image/jpeg")})
    data = r.json()
    for face in data.get("faces", []):
        print(f"  Name: {face.get('name')} (conf: {face.get('recognition_confidence')})")

    # Test reconstruction
    print("\n--- Reconstruction ---")
    with open(img_path, "rb") as fh:
        r = requests.post(f"{BASE}/reconstruct-face",
                          files={"file": ("test.jpg", fh, "image/jpeg")})
    rdata = r.json()
    print(f"  Success: {rdata.get('success')}")
    print(f"  Similarity: {rdata.get('similarity')}")
    print(f"  Method: {rdata.get('method')}")

    # Test privacy score
    print("\n--- Privacy Score ---")
    with open(img_path, "rb") as fh:
        r = requests.post(f"{BASE}/privacy-score",
                          files={"file": ("test.jpg", fh, "image/jpeg")})
    pdata = r.json()
    print(f"  Risk Level: {pdata.get('risk_level')}")
    print(f"  Overall Risk: {pdata.get('overall_risk')}")

    # Test spoof check
    print("\n--- Spoof Check ---")
    with open(img_path, "rb") as fh:
        r = requests.post(f"{BASE}/spoof-check",
                          files={"file": ("test.jpg", fh, "image/jpeg")})
    sdata = r.json()
    print(f"  Genuine: {sdata.get('is_genuine')}")
    print(f"  Confidence: {sdata.get('confidence')}")

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
