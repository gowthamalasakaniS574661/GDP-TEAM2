#!/usr/bin/env python3
"""Quick test script for the image_models and predict_quant endpoints"""
import subprocess
import time
import requests
import sys

# Start server
print("Starting Flask server...")
proc = subprocess.Popen(['python3', 'app.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

time.sleep(5)

try:
    # Test /api/image_models
    print("\n=== Testing /api/image_models ===")
    r = requests.get('http://127.0.0.1:5001/api/image_models')
    if r.status_code == 200:
        data = r.json()
        models = data.get('models', [])
        quant = [m for m in models if m.get('type') == 'quant']
        keras = [m for m in models if m.get('type') == 'keras']
        
        print(f"✓ Success! Found {len(models)} total models")
        print(f"  - {len(quant)} quant models (use /api/predict_quant)")
        print(f"  - {len(keras)} keras models (use /api/predict_image)")
        
        if quant:
            print(f"\nFirst 3 quant models (numeric features):")
            for m in quant[:3]:
                endpoint = m.get('endpoint', 'unknown')
                print(f"  • {m['name']} → {endpoint}")
        
        if keras:
            print(f"\nKeras models (image-based):")
            for m in keras:
                endpoint = m.get('endpoint', 'unknown')
                print(f"  • {m['name']} → {endpoint}")
    else:
        print(f"✗ Error: HTTP {r.status_code}")
        print(r.text)
    
    # Test /api/predict_quant
    print("\n=== Testing /api/predict_quant ===")
    r = requests.post('http://127.0.0.1:5001/api/predict_quant',
                      json={"model": "extratrees__ExtraTrees.joblib", "features": [1.0, 2.0, 3.0, 4.0]})
    if r.status_code == 200:
        result = r.json()
        print(f"✓ Success!")
        print(f"  Prediction: {result.get('prediction')}")
        print(f"  Confidence: {result.get('confidence'):.2%}")
    else:
        print(f"✗ Error: HTTP {r.status_code}")
        print(r.json())
        
finally:
    print("\nStopping server...")
    proc.terminate()
    proc.wait(timeout=2)
    print("Done!")
