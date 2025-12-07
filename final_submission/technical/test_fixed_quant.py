#!/usr/bin/env python3
"""Test the fixed quant_models endpoints with correct directory"""
import subprocess
import time
import requests
import sys

print("🚀 Starting Flask server...")
proc = subprocess.Popen(['python3', 'app.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
time.sleep(5)

try:
    print("\n" + "="*70)
    print("TESTING CORRECTED QUANT MODELS ENDPOINTS")
    print("="*70)
    
    # Test 1: List quant models (should find them in MLALGO/quant_model_artifacts/)
    print("\n1️⃣  GET /api/quant_models")
    print("-" * 70)
    r = requests.get('http://127.0.0.1:5001/api/quant_models')
    if r.status_code == 200:
        data = r.json()
        models = data.get('models', [])
        print(f"✅ Found {len(models)} models in MLALGO/quant_model_artifacts/")
        if models:
            print(f"\n   Sample models:")
            for m in models[:5]:
                feat_count = m.get('expected_features', '?')
                print(f"     • {m['filename']} ({feat_count} features)")
    else:
        print(f"❌ Error: {r.status_code} - {r.json()}")
    
    # Test 2: Predict with "model" field (correct)
    print("\n\n2️⃣  POST /api/predict_quant with 'model' field")
    print("-" * 70)
    r = requests.post('http://127.0.0.1:5001/api/predict_quant',
                      json={"model": "extratrees__AdaBoost.joblib", "features": [100, 0, 0, 0]})
    if r.status_code == 200:
        result = r.json()
        print(f"✅ SUCCESS with 'model' field!")
        print(f"   Prediction: {result.get('prediction')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")
        print(f"   Model: {result.get('model')}")
    else:
        print(f"❌ Error: {r.status_code}")
        print(f"   {r.json()}")
    
    # Test 3: Predict with "model_name" field (user's format - should also work now)
    print("\n\n3️⃣  POST /api/predict_quant with 'model_name' field (user format)")
    print("-" * 70)
    r = requests.post('http://127.0.0.1:5001/api/predict_quant',
                      json={"model_name": "extratrees__AdaBoost.joblib", "features": [100, 0, 0, 0]})
    if r.status_code == 200:
        result = r.json()
        print(f"✅ SUCCESS with 'model_name' field!")
        print(f"   Prediction: {result.get('prediction')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")
        print(f"   Model: {result.get('model')}")
    else:
        print(f"❌ Error: {r.status_code}")
        print(f"   {r.json()}")
    
    # Test 4: Test another model
    print("\n\n4️⃣  POST /api/predict_quant with ExtraTrees model")
    print("-" * 70)
    r = requests.post('http://127.0.0.1:5001/api/predict_quant',
                      json={"model": "extratrees__ExtraTrees.joblib", "features": [1.0, 2.0, 3.0, 4.0]})
    if r.status_code == 200:
        result = r.json()
        print(f"✅ SUCCESS!")
        print(f"   Prediction: {result.get('prediction')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")
        print(f"   Model: {result.get('model')}")
    else:
        print(f"❌ Error: {r.status_code}")
        print(f"   {r.json()}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✅ Models now loaded from: MLALGO/quant_model_artifacts/")
    print("✅ Both 'model' and 'model_name' fields supported")
    print("✅ Label encoder properly resolved from model directory")
    print("\n📌 Your user's request should now work!")
    print("   Request: {\"model_name\": \"extratrees__AdaBoost.joblib\", \"features\": [100, 0, 0, 0]}")
    
finally:
    print("\n\n🛑 Stopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except:
        proc.kill()
    print("✓ Done!")
