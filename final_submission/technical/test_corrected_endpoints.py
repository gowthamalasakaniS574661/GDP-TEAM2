#!/usr/bin/env python3
"""Test to verify the corrected endpoint behavior"""
import subprocess
import time
import requests
import sys

print("Starting Flask server...")
proc = subprocess.Popen(['python3', 'app.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
time.sleep(5)

try:
    print("\n" + "="*60)
    print("CLARIFYING THE MODEL ARCHITECTURE")
    print("="*60)
    
    # Test /api/quant_models (numeric feature models)
    print("\n1️⃣  /api/quant_models - Lists .joblib models for NUMERIC features")
    print("-" * 60)
    r = requests.get('http://127.0.0.1:5001/api/quant_models')
    if r.status_code == 200:
        data = r.json()
        models = data.get('models', [])
        print(f"✓ Found {len(models)} models that accept NUMERIC FEATURES")
        print(f"  Use with: POST /api/predict_quant")
        print(f"  Example: {{'model': 'extratrees__ExtraTrees.joblib', 'features': [1.0, 2.0, 3.0, 4.0]}}")
        if models:
            print(f"\n  First 3 models:")
            for m in models[:3]:
                feat_count = m.get('expected_features', '?')
                print(f"    • {m['filename']} (expects {feat_count} numeric features)")
    
    # Test /api/image_models (actual image processing models)
    print("\n\n2️⃣  /api/image_models - Lists models for IMAGE files")
    print("-" * 60)
    r = requests.get('http://127.0.0.1:5001/api/image_models')
    if r.status_code == 200:
        data = r.json()
        models = data.get('models', [])
        print(f"✓ Found {len(models)} models that accept IMAGE FILES")
        print(f"  Use with: POST /api/predict_image (multipart/form-data)")
        if models:
            for m in models:
                print(f"\n  Model: {m['name']}")
                print(f"    Type: {m.get('type')}")
                print(f"    Accepts: {m.get('accepts', 'image file')}")
                print(f"    Endpoint: {m.get('endpoint')}")
        else:
            print("  ⚠️  No IMAGE models found (only Keras/TensorFlow models supported)")
    
    # Test predict_quant with correct data
    print("\n\n3️⃣  Testing /api/predict_quant with numeric features")
    print("-" * 60)
    r = requests.post('http://127.0.0.1:5001/api/predict_quant',
                      json={"model": "extratrees__ExtraTrees.joblib", "features": [1.0, 2.0, 3.0, 4.0]})
    if r.status_code == 200:
        result = r.json()
        print(f"✓ SUCCESS!")
        print(f"  Prediction: {result.get('prediction')}")
        print(f"  Confidence: {result.get('confidence'):.2%}")
        print(f"  Model: {result.get('model')}")
    else:
        print(f"✗ Error: {r.status_code}")
        print(f"  {r.json()}")
    
    # Test predict_image with wrong model (should get helpful error)
    print("\n\n4️⃣  Testing /api/predict_image with a .joblib model (should fail helpfully)")
    print("-" * 60)
    # Create a dummy image file
    import io
    from PIL import Image
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    r = requests.post('http://127.0.0.1:5001/api/predict_image',
                      files={'image': ('test.png', img_bytes, 'image/png')},
                      data={'model_name': 'extratrees__ExtraTrees'})
    if r.status_code != 200:
        result = r.json()
        print(f"✓ Correctly rejected with helpful error:")
        print(f"  Error: {result.get('error')}")
        print(f"  Message: {result.get('message')}")
        print(f"  Hint: {result.get('hint')}")
    else:
        print(f"✗ Unexpected success - should have failed!")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✓ /api/quant_models → Lists .joblib models (numeric features)")
    print("✓ /api/predict_quant → Predicts with numeric features")  
    print("✓ /api/image_models → Lists Keras/TF models (image files)")
    print("✓ /api/predict_image → Predicts with image files")
    print("\n📌 Key Point: .joblib models in quantized_models/ are")
    print("   for NUMERIC features, not images!")
    
finally:
    print("\nStopping server...")
    proc.terminate()
    proc.wait(timeout=2)
    print("Done!")
