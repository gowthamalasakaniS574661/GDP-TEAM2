# app.py  -- Option 1: only expose compatible models (matching *_tfidf.joblib)
from flask import Flask, request, jsonify, render_template
import os, csv, joblib, traceback
import numpy as np
import json

app = Flask(__name__)

ART_DIR = os.path.join("MLALGO", "text_model_artifacts")
LABEL_PATH = os.path.join(ART_DIR, "label_encoder.joblib")
SUMMARY_CSV = os.path.join(ART_DIR, "training_summary_text_models.csv")
# allow overriding the port via environment variable PORT or DEFAULT_PORT
DEFAULT_PORT = int(os.getenv('PORT', os.getenv('DEFAULT_PORT', '5001')))

# directory for quantized numeric feature models (the CORRECT location)
QUANT_MODELS_DIR = os.environ.get('QUANT_MODELS_DIR', os.path.join('MLALGO', 'quant_model_artifacts'))
# directory for local image (quantized) models - OLD LOCATION, DEPRECATED
IMAGE_MODELS_DIR = os.environ.get('IMAGE_MODELS_DIR', os.path.join('MLALGO', 'image_model_artifacts', 'quantized_models'))
# alternate image model layouts (user-provided)
IMAGE_MODELS_DIR_ALT = os.path.join('MLALGO', 'image_models')
IMAGE_MODELS_DIR_ALL = os.path.join('MLALGO', 'image_models_all')  # NEW: Additional models directory
_image_models_cache = {'ts': 0.0, 'data': None, 'ttl': float(os.getenv('IMAGE_MODELS_CACHE_TTL', '10'))}
# optional warm-load of Keras models at discovery time (requires TensorFlow installed). Default off.
IMAGE_MODELS_WARM_LOAD = os.getenv('IMAGE_MODELS_WARM_LOAD', 'false').lower() in ('1','true','yes')

# prediction behavior tuning (configurable via env var)
CONFIDENCE_THRESHOLD = float(os.getenv('CONF_THRESHOLD', '0.60'))  # below this value we return a low-confidence response

import itertools
# lazy import helpers for image models (kept optional so text endpoints don't require heavy deps)
try:
    from MLALGO.src import image_model_loader, image_processing
except Exception:
    image_model_loader = None
    image_processing = None

# ---------------- utilities ----------------
def safe_list_joblib_files():
    """List model files (exclude vectorizers and label encoder)."""
    if not os.path.isdir(ART_DIR):
        return []
    files = sorted([f for f in os.listdir(ART_DIR) if f.endswith(".joblib")])
    # exclude vectorizers and label encoder
    files = [f for f in files if not (f.endswith("_tfidf.joblib") or f == "tfidf.joblib" or f == os.path.basename(LABEL_PATH))]
    return files

def list_vectorizers():
    """Return vectorizer filenames present (tfidf or *_tfidf)."""
    if not os.path.isdir(ART_DIR):
        return []
    return sorted([f for f in os.listdir(ART_DIR) if f.endswith("tfidf.joblib")])

def parse_summary_csv():
    """Robust CSV parse supporting fs,model,acc,time_s,fs_shape header."""
    meta = {}
    if not os.path.exists(SUMMARY_CSV):
        return meta
    try:
        with open(SUMMARY_CSV, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = [[c.strip() for c in r] for r in reader if any(cell.strip() for cell in r)]
            if not rows:
                return meta
            header = rows[0]
            header_lower = [h.lower() for h in header]
            # determine indices
            idx = {'fs': None, 'model': None, 'acc': None, 'time_s': None, 'fs_shape': None}
            for i,h in enumerate(header_lower):
                if h in ('fs','feature_selector','feature'):
                    idx['fs'] = i
                if h in ('model','algorithm','alg'):
                    idx['model'] = i
                if h in ('acc','accuracy'):
                    idx['acc'] = i
                if h in ('time_s','runtime','time'):
                    idx['time_s'] = i
                if h in ('fs_shape','shape'):
                    idx['fs_shape'] = i
            # assume header exists if fs and model detected
            has_header = idx['fs'] is not None and idx['model'] is not None
            data_rows = rows[1:] if has_header else rows
            for r in data_rows:
                row = list(r)
                if len(row) >= 6 and row[0].isdigit():
                    row = row[1:]
                if has_header:
                    fs = row[idx['fs']].strip() if idx['fs'] is not None and idx['fs'] < len(row) else ''
                    model = row[idx['model']].strip() if idx['model'] is not None and idx['model'] < len(row) else ''
                    acc = row[idx['acc']].strip() if idx['acc'] is not None and idx['acc'] < len(row) else ''
                    time_s = row[idx['time_s']].strip() if idx['time_s'] is not None and idx['time_s'] < len(row) else ''
                    shape = row[idx['fs_shape']].strip() if idx['fs_shape'] is not None and idx['fs_shape'] < len(row) else ''
                else:
                    fs = row[0] if len(row) > 0 else ''
                    model = row[1] if len(row) > 1 else ''
                    acc = row[2] if len(row) > 2 else ''
                    time_s = row[3] if len(row) > 3 else ''
                    shape = row[4] if len(row) > 4 else ''
                if fs and model:
                    filename = f"{fs}__{model}.joblib"
                    meta[filename] = {
                        "feature_selector": fs,
                        "algorithm": model,
                        "accuracy": acc,
                        "runtime": time_s,
                        "shape": shape
                    }
    except Exception as e:
        print("Error parsing CSV:", e)
        traceback.print_exc()
    return meta

_summary_meta = parse_summary_csv()

# Simple in-memory cache for quant model discovery to avoid repeated heavy joblib loads.
# Cache structure: {'ts': float_timestamp, 'data': list_of_models}
_quant_models_cache = {'ts': 0.0, 'data': None, 'ttl': float(os.getenv('QUANT_MODELS_CACHE_TTL', '10'))}

# try load label encoder (optional)
_label_encoder = None
_label_encoder_path = LABEL_PATH

def get_label_encoder():
    """Lazily load and cache the label encoder to avoid heavy imports at startup."""
    global _label_encoder
    if _label_encoder is not None:
        return _label_encoder
    try:
        if os.path.exists(_label_encoder_path):
            _label_encoder = joblib.load(_label_encoder_path)
            print("Label encoder loaded (lazy).")
            return _label_encoder
    except Exception as e:
        # don't raise here; just return None and let callers handle absence
        print("Could not load label encoder lazily:", e)
    return None

# ---------------- API: models (filtered by available vectorizers) ----------------
@app.route("/api/models", methods=["GET"])
def api_models():
    try:
        available_vecs = list_vectorizers()  # e.g. ['tfidf.joblib', 'variance_tfidf.joblib', ...]
        # derive available prefixes (e.g. 'variance' from 'variance_tfidf.joblib', include '' for generic tfidf)
        prefixes = set()
        for v in available_vecs:
            if v == "tfidf.joblib":
                prefixes.add("")  # generic fallback
            else:
                prefixes.add(v.replace('_tfidf.joblib','').replace('__tfidf.joblib',''))
        all_models = safe_list_joblib_files()
        # keep only models whose prefix has a matching vectorizer
        compatible_models = []
        for m in all_models:
            prefix = m.split("__")[0]
            if prefix in prefixes or "" in prefixes:
                # include
                meta = _summary_meta.get(m, {})
                feat = meta.get("feature_selector","") or prefix
                algo = meta.get("algorithm","") or m.split("__")[1].replace('.joblib','') if "__" in m else m.replace('.joblib','')
                acc = meta.get("accuracy","")
                runtime = meta.get("runtime","")
                shape = meta.get("shape","")
                expected_vec = f"{prefix}_tfidf.joblib"
                # if expected_vec doesn't exist but generic tfidf exists, mark expected as 'tfidf.joblib'
                compatible = (expected_vec in available_vecs) or ("tfidf.joblib" in available_vecs)
                if not compatible:
                    # skip if not compatible (safety)
                    continue
                compatible_models.append({
                    "filename": m,
                    "feature_selector": feat,
                    "algorithm": algo,
                    "accuracy": acc,
                    "runtime": runtime,
                    "shape": shape,
                    "expected_vectorizer": expected_vec,
                    "compatible": compatible
                })
        # include server-side confidence threshold so the UI can show helpful tooltips
        return jsonify({"models": compatible_models, "available_vectorizers": available_vecs, "confidence_threshold": CONFIDENCE_THRESHOLD})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"models": [], "available_vectorizers": list_vectorizers(), "error": str(e), "confidence_threshold": CONFIDENCE_THRESHOLD}), 500

# ---------------- Helper: load model & vectorizer ----------------
def load_vectorizer_for_model(model_filename):
    prefix = model_filename.split("__")[0]
    expected = f"{prefix}_tfidf.joblib"
    expected_path = os.path.join(ART_DIR, expected)
    if os.path.exists(expected_path):
        return joblib.load(expected_path), expected
    # fallback to generic tfidf.joblib if present
    generic = os.path.join(ART_DIR, "tfidf.joblib")
    if os.path.exists(generic):
        return joblib.load(generic), "tfidf.joblib"
    raise FileNotFoundError(f"No vectorizer found for model {model_filename} (expected {expected} or tfidf.joblib)")


def load_selector_for_model_if_present(model_filename):
    """Try to find and load a feature-selector artifact that matches the model prefix.
    Returns (selector, selector_filename) or (None, None) if not found.
    Common names tried: <prefix>_selector.joblib, <prefix>_kbest.joblib, <prefix>_fs.joblib
    """
    prefix = model_filename.split("__")[0]
    candidates = [
        f"{prefix}_selector.joblib",
        f"{prefix}_kbest.joblib",
        f"{prefix}_fs.joblib",
        f"{prefix}_sel.joblib",
        f"{prefix}_selectkbest.joblib",
        f"{prefix}_selectk.joblib",
    ]
    for c in candidates:
        p = os.path.join(ART_DIR, c)
        if os.path.exists(p):
            try:
                sel = joblib.load(p)
                return sel, c
            except Exception:
                # ignore load errors and continue
                continue
    return None, None


def is_gibberish(text: str) -> bool:
    """Lightweight heuristic to detect gibberish / unhelpful inputs.
    Returns True when the input should be rejected before prediction.
    Heuristics used (fast):
      - very short strings (<5 chars)
      - very low alphabetic ratio
      - single-token strings with very few vowels
      - extremely long repeated-character runs
    These are conservative and are intended as a first-pass filter.
    """
    if not text:
        return True
    s = str(text).strip()
    if len(s) < 5:
        return True
    alpha_chars = [c for c in s if c.isalpha()]
    if len(alpha_chars) / max(1, len(s)) < 0.5:
        return True
    # overall vowel ratio across all alphabetic characters (helps catch strings of consonants)
    letters_all = [c for c in s.lower() if c.isalpha()]
    if letters_all:
        vowels_all = sum(1 for c in letters_all if c in 'aeiou')
        vowel_ratio_all = vowels_all / len(letters_all)
        # if overall vowel ratio is extremely low, likely gibberish (e.g. HCVGHS JHCHG...)
        if vowel_ratio_all < 0.15:
            return True
    words = s.split()
    # require at least two short words to be considered a valid symptom phrase
    if len(words) < 2:
        # allow single word only if it looks like a real word (has vowels)
        if len(words) == 1:
            letters = [c for c in s.lower() if c.isalpha()]
            if not letters:
                return True
            vowels = sum(1 for c in letters if c in 'aeiou')
            vowel_ratio = vowels / len(letters)
            if vowel_ratio < 0.2:
                return True
        else:
            return True
    if len(words) == 1:
        letters = [c for c in s.lower() if c.isalpha()]
        if not letters:
            return True
        vowels = sum(1 for c in letters if c in 'aeiou')
        vowel_ratio = vowels / len(letters)
        # unlikely to be a natural word if vowel ratio is tiny
        if vowel_ratio < 0.2:
            return True
    # detect long consonant runs inside any token (e.g. 'JHCHGKXBKJSAXHJ')
    for w in words:
        # consonant runs (no vowels) longer than 5 are suspicious
        max_run = 0
        cur_run = 0
        for ch in w.lower():
            if ch.isalpha() and ch not in 'aeiou':
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        if max_run >= 5:
            return True
        # detect huge repeated runs (e.g. aaaaaaaa)
        runs = [len(list(g)) for _, g in itertools.groupby(s)]
        if runs and max(runs) > max(3, len(s) * 0.6):
            return True
    return False


def compute_confidence(model, X):
    """Return a confidence score in [0,1] for the sample X (1-row).
    Preference order:
      - predict_proba (native)
      - decision_function -> softmax over margins (fallback for LinearSVC, SVM, etc.)
      - None if cannot compute
    """
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[0]
            return float(max(probs))
    except Exception:
        pass
    try:
        if hasattr(model, 'decision_function'):
            margins = model.decision_function(X)
            # ensure 1D array of class scores
            if np.ndim(margins) == 0:
                margins = np.array([margins])
            if np.ndim(margins) == 2:
                margins = margins[0]
            # numeric stability softmax
            exps = np.exp(margins - np.max(margins))
            probs = exps / np.sum(exps)
            return float(np.max(probs))
    except Exception:
        pass
    return None


# ---------------- API: predict_image (quantized / image models) ----------------
@app.route("/api/predict_image", methods=["POST"])
def api_predict_image():
    # This endpoint is intentionally lightweight: it will only work when the optional runtimes
    # required by the quantized models are available (onnxruntime, tflite-runtime/tensorflow, torch).
    try:
        # image_processing is required for preprocessing; image_model_loader is optional if we have alternate keras-style models
        if image_processing is None:
            return jsonify({"error": "runtimes_missing", "message": "Image processing helpers not available in this Python environment."}), 501

        # require a file named 'image'
        if 'image' not in request.files:
            return jsonify({"error": "no_image", "message": "Upload an image file using form field 'image'."}), 400
        file = request.files['image']
        requested_model = request.form.get('model_name') or request.args.get('model_name')

        # discover quant/image adapters from the configured directories
        models = {}
        
        # NOTE: We do NOT scan IMAGE_MODELS_DIR for .joblib files here because those
        # are numeric feature models, not image models. If user requests them, we'll
        # return a helpful error message directing them to /api/predict_quant.
        
        # 1) Advanced quantized IMAGE models (onnx/tflite/torch) if loader available
        # These are actual image processing models (not the .joblib sklearn models)
        if image_model_loader is not None and os.path.isdir(IMAGE_MODELS_DIR):
            try:
                qm = image_model_loader.discover_models(IMAGE_MODELS_DIR)
                # Only include models that are actual image processors (ONNX/TFLite/Torch)
                models.update(qm)
            except Exception:
                pass

        # 2) discover Keras-style models under MLALGO/image_models/<name>/model/
        # Also handle top-level layout: MLALGO/image_models/model/ (no subfolder)
        try:
            if os.path.isdir(IMAGE_MODELS_DIR_ALT):
                # First check if there's a top-level 'model' directory (no subfolder nesting)
                top_model_dir = os.path.join(IMAGE_MODELS_DIR_ALT, 'model')
                entries_to_process = []
                
                if os.path.isdir(top_model_dir):
                    # Top-level model exists: MLALGO/image_models/model/
                    # Register as "image_models" (basename of IMAGE_MODELS_DIR_ALT)
                    entries_to_process.append((os.path.basename(IMAGE_MODELS_DIR_ALT), IMAGE_MODELS_DIR_ALT, top_model_dir))
                else:
                    # Standard layout: MLALGO/image_models/<name>/model/
                    for entry in sorted(os.listdir(IMAGE_MODELS_DIR_ALT)):
                        sub = os.path.join(IMAGE_MODELS_DIR_ALT, entry)
                        if not os.path.isdir(sub):
                            continue
                        model_sub = os.path.join(sub, 'model')
                        if not os.path.isdir(model_sub):
                            continue
                        entries_to_process.append((entry, sub, model_sub))
                
                # Process all discovered entries
                for entry, sub, model_sub in entries_to_process:
                    # prefer best_model.keras, then any .h5/.keras, then a SavedModel directory
                    found = None
                    cand = os.path.join(model_sub, 'best_model.keras')
                    if os.path.exists(cand):
                        found = cand
                    else:
                        for fn in os.listdir(model_sub):
                            if fn.lower().endswith('.h5') or fn.lower().endswith('.keras'):
                                found = os.path.join(model_sub, fn)
                                break
                        if found is None and os.path.exists(os.path.join(model_sub, 'saved_model.pb')):
                            found = model_sub
                    if not found:
                        continue

                    # create a lazy Keras adapter
                    class KerasAdapter:
                        def __init__(self, path, classes_path=None, fs_saved_dir=None):
                            self.path = path
                            self.classes_path = classes_path
                            self.fs_saved_dir = fs_saved_dir
                            self.model = None
                            self.labels = None
                            self.preprocessor = None

                        def load_preprocessor(self):
                            if not self.fs_saved_dir or self.preprocessor is not None:
                                return
                            try:
                                # look for common artifact names
                                candidates = ['preprocessor.joblib', 'preprocess.joblib', 'scaler.joblib', 'preprocessor.pkl', 'preprocess.pkl']
                                for c in candidates:
                                    p = os.path.join(self.fs_saved_dir, c)
                                    if os.path.exists(p):
                                        try:
                                            self.preprocessor = joblib.load(p)
                                            return
                                        except Exception:
                                            # try pickle fallback
                                            try:
                                                import pickle
                                                with open(p, 'rb') as pf:
                                                    self.preprocessor = pickle.load(pf)
                                                    return
                                            except Exception:
                                                continue
                                # if no named candidate found, try any joblib/pkl in fs_saved_dir
                                for fn in os.listdir(self.fs_saved_dir or '.'):
                                    ln = fn.lower()
                                    if ln.endswith('.joblib') or ln.endswith('.pkl') or ln.endswith('.pickle'):
                                        p = os.path.join(self.fs_saved_dir, fn)
                                        try:
                                            self.preprocessor = joblib.load(p)
                                            return
                                        except Exception:
                                            try:
                                                import pickle
                                                with open(p, 'rb') as pf:
                                                    self.preprocessor = pickle.load(pf)
                                                    return
                                            except Exception:
                                                continue
                            except Exception:
                                self.preprocessor = None

                        def load(self):
                            if self.model is not None:
                                return
                            try:
                                import tensorflow as tf
                            except Exception as e:
                                raise RuntimeError('tensorflow not installed') from e
                            # load model (supports HDF5 file or SavedModel dir)
                            self.model = tf.keras.models.load_model(self.path)
                            # try load class indices if present
                            if self.classes_path and os.path.exists(self.classes_path):
                                try:
                                    with open(self.classes_path, 'r', encoding='utf-8') as cf:
                                        data = json.load(cf)
                                        # common format is {class_name: index}
                                        if isinstance(data, dict):
                                            # invert to list index->label
                                            inv = [None] * (max(data.values()) + 1)
                                            for k,v in data.items():
                                                try:
                                                    inv[int(v)] = k
                                                except Exception:
                                                    pass
                                            self.labels = inv
                                        elif isinstance(data, list):
                                            self.labels = data
                                except Exception:
                                    self.labels = None
                            # attempt to load preprocessor if provided
                            try:
                                self.load_preprocessor()
                            except Exception:
                                self.preprocessor = None

                        def predict(self, image_np):
                            # image_np expected (1,H,W,3) float32
                            self.load()
                            arr = image_np.astype('float32')
                            # if a preprocessor is available, try to apply it
                            if self.preprocessor is not None:
                                try:
                                    # prefer transform method
                                    if hasattr(self.preprocessor, 'transform'):
                                        try:
                                            arr2 = self.preprocessor.transform(arr)
                                        except Exception:
                                            # try flatten for sklearn-style transformers
                                            try:
                                                arr2 = self.preprocessor.transform(arr.reshape((arr.shape[0], -1)))
                                            except Exception:
                                                raise
                                    elif callable(self.preprocessor):
                                        arr2 = self.preprocessor(arr)
                                    else:
                                        arr2 = arr
                                    # ensure numpy array
                                    import numpy as _np
                                    arr2 = _np.asarray(arr2)
                                    # if preprocessor reduced to 1D features and model expects image, try to reshape back heuristically
                                    out = self.model.predict(arr2)
                                    return out
                                except Exception:
                                    # swallow preprocessor errors and fallback to direct model.predict
                                    pass
                            out = self.model.predict(arr)
                            return out

                    classes_path = os.path.join(sub, 'model', 'class_indices.json')
                    fs_saved_dir = os.path.join(sub, 'fs_saved') if os.path.isdir(os.path.join(sub, 'fs_saved')) else None
                    models[entry] = KerasAdapter(found, classes_path, fs_saved_dir)
        except Exception:
            # ignore alt discovery errors
            pass

        # 3) NEW: Discover standalone Keras models in IMAGE_MODELS_DIR_ALL
        # This directory contains .keras/.h5 files directly
        try:
            if os.path.isdir(IMAGE_MODELS_DIR_ALL):
                for filename in sorted(os.listdir(IMAGE_MODELS_DIR_ALL)):
                    if filename.lower().endswith('.keras') or filename.lower().endswith('.h5'):
                        model_path = os.path.join(IMAGE_MODELS_DIR_ALL, filename)
                        if os.path.isfile(model_path):
                            model_name = os.path.splitext(filename)[0]
                            
                            # Look for metadata/class_indices
                            classes_path = os.path.join(IMAGE_MODELS_DIR_ALL, model_name + '_class_indices.json')
                            if not os.path.exists(classes_path):
                                classes_path = None
                            
                            # Create a simple Keras adapter for standalone models
                            class SimpleKerasAdapter:
                                def __init__(self, path, classes_path=None):
                                    self.path = path
                                    self.classes_path = classes_path
                                    self.model = None
                                    self.labels = None

                                def load(self):
                                    if self.model is not None:
                                        return
                                    try:
                                        import tensorflow as tf
                                    except Exception as e:
                                        raise RuntimeError('tensorflow not installed') from e
                                    self.model = tf.keras.models.load_model(self.path)
                                    
                                    # Load class indices if present
                                    if self.classes_path and os.path.exists(self.classes_path):
                                        try:
                                            with open(self.classes_path, 'r', encoding='utf-8') as cf:
                                                data = json.load(cf)
                                                if isinstance(data, dict):
                                                    inv = [None] * (max(data.values()) + 1)
                                                    for k,v in data.items():
                                                        try:
                                                            inv[int(v)] = k
                                                        except Exception:
                                                            pass
                                                    self.labels = inv
                                                elif isinstance(data, list):
                                                    self.labels = data
                                        except Exception:
                                            self.labels = None

                                def predict(self, image_np):
                                    self.load()
                                    return self.model.predict(image_np.astype('float32'))
                            
                            models[model_name] = SimpleKerasAdapter(model_path, classes_path)
        except Exception as e:
            print(f"[DEBUG] Error discovering models from image_models_all in predict: {e}")
            pass

        if not models:
            return jsonify({
                "error": "no_models", 
                "message": f"No image-processing models found.",
                "hint": "This endpoint requires actual image models (Keras/TensorFlow, ONNX, TFLite). The .joblib models in quantized_models/ are for numeric features - use /api/predict_quant instead.",
                "available_numeric_models_endpoint": "/api/quant_models"
            }), 404

        if requested_model:
            adapter = models.get(requested_model)
            if adapter is None:
                # Check if it's a .joblib model that was mistakenly requested
                joblib_path = os.path.join(IMAGE_MODELS_DIR, requested_model + '.joblib')
                if os.path.exists(joblib_path):
                    return jsonify({
                        "error": "wrong_endpoint",
                        "message": f"Model '{requested_model}' is a .joblib model that expects numeric features, not images.",
                        "hint": f"Use POST /api/predict_quant with JSON: {{\"model\": \"{requested_model}.joblib\", \"features\": [...]}}",
                        "redirect_to": "/api/predict_quant"
                    }), 400
                return jsonify({"error": "model_not_found", "message": f"Image model '{requested_model}' not found", "available_models": list(models.keys())}), 404
            model_name = requested_model
        else:
            # pick the first discovered model
            model_name, adapter = next(iter(models.items()))

        # preprocess image (default target size 224x224). If you have model-specific metadata,
        # put a sidecar JSON file next to the model with input_size [H,W] to override.
        target_size = (224, 224)
        # check for sidecar metadata
        meta_path = os.path.join(IMAGE_MODELS_DIR, model_name + '.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as mf:
                    mm = json.load(mf)
                    if isinstance(mm.get('input_size'), (list, tuple)) and len(mm['input_size']) == 2:
                        target_size = tuple(mm['input_size'])
            except Exception:
                pass

        img = image_processing.load_image_file(file, target_size=target_size)

        out = adapter.predict(img)
        # normalize outputs to probabilities when possible
        logits = out[0] if getattr(out, 'ndim', 0) == 2 else out
        # guard numeric shapes
        logits = np.asarray(logits).ravel()
        exps = np.exp(logits - np.max(logits)) if logits.size > 0 else np.array([1.0])
        probs = exps / np.sum(exps)
        pred_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))

        # determine label mapping. KerasAdapter may provide 'labels' attribute (list),
        # otherwise look for a sidecar JSON next to models in IMAGE_MODELS_DIR
        pred_label = str(pred_idx)
        try:
            # adapter-provided labels (e.g., KerasAdapter sets .labels)
            if hasattr(adapter, 'labels') and adapter.labels:
                lab = adapter.labels
                if isinstance(lab, list) and pred_idx < len(lab):
                    pred_label = lab[pred_idx]
                elif isinstance(lab, dict):
                    pred_label = lab.get(str(pred_idx), str(pred_idx))
            else:
                # attempt to read conventional sidecar next to IMAGE_MODELS_DIR (for quantized adapters)
                label_map = None
                label_path = os.path.join(IMAGE_MODELS_DIR, model_name + '.json')
                if os.path.exists(label_path):
                    try:
                        with open(label_path, 'r', encoding='utf-8') as f:
                            label_map = json.load(f).get('labels')
                    except Exception:
                        label_map = None
                
                # If no labels found yet, try fallback to shared class_indices.json
                if not label_map:
                    # Try the original image_models class_indices.json as fallback
                    fallback_path = os.path.join(IMAGE_MODELS_DIR_ALT, 'model', 'class_indices.json')
                    if os.path.exists(fallback_path):
                        try:
                            with open(fallback_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                # Convert {class_name: index} to [class_name, ...]
                                if isinstance(data, dict):
                                    inv = [None] * (max(data.values()) + 1) if data else []
                                    for k, v in data.items():
                                        try:
                                            inv[int(v)] = k
                                        except Exception:
                                            pass
                                    label_map = inv
                                elif isinstance(data, list):
                                    label_map = data
                        except Exception:
                            pass
                
                if isinstance(label_map, dict):
                    pred_label = label_map.get(str(pred_idx), str(pred_idx))
                elif isinstance(label_map, list) and pred_idx < len(label_map):
                    pred_label = label_map[pred_idx]
        except Exception:
            pred_label = str(pred_idx)

        return jsonify({
            "prediction": pred_label,
            "prediction_index": pred_idx,
            "confidence": confidence,
            "model": model_name
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "prediction_failed", "message": str(e)}), 500


@app.route("/api/image_models", methods=["GET"])
def api_image_models():
    """Enumerate available IMAGE models (models that accept image files as input).
    NOTE: This endpoint only returns models that can process images (Keras/TensorFlow models).
    For numeric feature models (.joblib), use /api/quant_models instead.
    Returns: {models: [{name,type,input_size,labels}]}
    """
    try:
        results = []
        now = float(__import__('time').time())
        cache = _image_models_cache
        print(f"[DEBUG api_image_models] Called. Cache data: {cache.get('data') is not None}, ts: {cache.get('ts', 0.0)}, now: {now}, ttl: {cache.get('ttl')}")
        if cache.get('data') is not None and (now - float(cache.get('ts', 0.0))) < float(cache.get('ttl', 10.0)):
            print(f"[DEBUG api_image_models] Returning cached result with {len(cache.get('data', []))} models")
            return jsonify({"models": cache['data'], "cached": True, "ttl": cache['ttl']})

        # NOTE: We do NOT scan IMAGE_MODELS_DIR (.joblib files) here because those models
        # expect numeric features, not images. They should be accessed via /api/quant_models.
        
        # Discover Keras-style image models under MLALGO/image_models/<name>/model/
        # NOTE: The user has a top-level layout: MLALGO/image_models/{model/, fs_saved/}
        # This is a single Keras model, not a collection of subfolders.
        # We'll register it as a single "keras" model named "image_models" (or basename of the dir).
        try:
            if os.path.isdir(IMAGE_MODELS_DIR_ALT):
                top_model_sub = os.path.join(IMAGE_MODELS_DIR_ALT, 'model')
                if os.path.isdir(top_model_sub):
                    # This is a single top-level Keras model layout
                    entry = os.path.basename(os.path.normpath(IMAGE_MODELS_DIR_ALT)) or 'image_model'
                    sub = IMAGE_MODELS_DIR_ALT
                    model_sub = top_model_sub
                    labels = None
                    input_size = None
                    classes_path = os.path.join(model_sub, 'class_indices.json')
                    if os.path.exists(classes_path):
                        try:
                            with open(classes_path, 'r', encoding='utf-8') as cf:
                                data = json.load(cf)
                                if isinstance(data, list):
                                    labels = [str(x) for x in data]
                                elif isinstance(data, dict):
                                    inv = [None] * (max(data.values()) + 1) if data else None
                                    if inv is not None:
                                        for k2, v2 in data.items():
                                            try:
                                                inv[int(v2)] = str(k2)
                                            except Exception:
                                                pass
                                        labels = inv
                        except Exception:
                            labels = None
                    meta_path = os.path.join(model_sub, 'model.json')
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, 'r', encoding='utf-8') as mf:
                                md = json.load(mf)
                                if isinstance(md.get('input_size'), (list, tuple)) and len(md.get('input_size')) == 2:
                                    input_size = list(md.get('input_size'))
                        except Exception:
                            pass
                    results.append({"name": entry, "type": "keras", "input_size": input_size, "labels": labels, "has_fs_saved": os.path.isdir(os.path.join(sub, 'fs_saved')), "endpoint": "/api/predict_image", "accepts": "image file"})
        except Exception:
            pass

        # NEW: Discover standalone Keras models in IMAGE_MODELS_DIR_ALL
        # This directory contains .keras files directly without subdirectories
        try:
            if os.path.isdir(IMAGE_MODELS_DIR_ALL):
                for filename in sorted(os.listdir(IMAGE_MODELS_DIR_ALL)):
                    if filename.lower().endswith('.keras') or filename.lower().endswith('.h5'):
                        model_path = os.path.join(IMAGE_MODELS_DIR_ALL, filename)
                        if os.path.isfile(model_path):
                            # Use filename without extension as model name
                            model_name = os.path.splitext(filename)[0]
                            
                            # Try to load metadata if available
                            meta_path = os.path.join(IMAGE_MODELS_DIR_ALL, model_name + '.json')
                            labels = None
                            input_size = None
                            
                            if os.path.exists(meta_path):
                                try:
                                    with open(meta_path, 'r', encoding='utf-8') as mf:
                                        md = json.load(mf)
                                        if isinstance(md.get('input_size'), (list, tuple)) and len(md.get('input_size')) == 2:
                                            input_size = list(md.get('input_size'))
                                        if isinstance(md.get('labels'), list):
                                            labels = md.get('labels')
                                except Exception:
                                    pass
                            
                            # Check for class_indices.json
                            classes_path = os.path.join(IMAGE_MODELS_DIR_ALL, model_name + '_class_indices.json')
                            if not labels and os.path.exists(classes_path):
                                try:
                                    with open(classes_path, 'r', encoding='utf-8') as cf:
                                        data = json.load(cf)
                                        if isinstance(data, list):
                                            labels = [str(x) for x in data]
                                        elif isinstance(data, dict):
                                            inv = [None] * (max(data.values()) + 1) if data else None
                                            if inv is not None:
                                                for k2, v2 in data.items():
                                                    try:
                                                        inv[int(v2)] = str(k2)
                                                    except Exception:
                                                        pass
                                                labels = inv
                                except Exception:
                                    pass
                            
                            # FALLBACK: If no labels found, use shared class_indices from original model
                            if not labels:
                                fallback_path = os.path.join(IMAGE_MODELS_DIR_ALT, 'model', 'class_indices.json')
                                if os.path.exists(fallback_path):
                                    try:
                                        with open(fallback_path, 'r', encoding='utf-8') as cf:
                                            data = json.load(cf)
                                            if isinstance(data, list):
                                                labels = [str(x) for x in data]
                                            elif isinstance(data, dict):
                                                inv = [None] * (max(data.values()) + 1) if data else []
                                                for k2, v2 in data.items():
                                                    try:
                                                        inv[int(v2)] = str(k2)
                                                    except Exception:
                                                        pass
                                                labels = inv
                                    except Exception:
                                        pass
                            
                            results.append({
                                "name": model_name,
                                "type": "keras",
                                "input_size": input_size,
                                "labels": labels,
                                "source": "image_models_all",
                                "endpoint": "/api/predict_image",
                                "accepts": "image file"
                            })
        except Exception as e:
            print(f"[DEBUG] Error discovering models from image_models_all: {e}")
            pass

        # optional warm-load verification for keras models (if enabled)
        if IMAGE_MODELS_WARM_LOAD:
            try:
                import tensorflow as _tf
                for r in results:
                    if r.get('type') == 'keras':
                        # try to locate model path
                        name = r.get('name')
                        # check top-level
                        top_model = os.path.join(IMAGE_MODELS_DIR_ALT, 'model')
                        if os.path.isdir(top_model) and (os.path.basename(os.path.normpath(IMAGE_MODELS_DIR_ALT)) == name):
                            mpath = top_model
                        else:
                            mpath = os.path.join(IMAGE_MODELS_DIR_ALT, name, 'model')
                        try:
                            # attempt to load (short-circuit heavy memory by using load_model)
                            _tf.keras.models.load_model(mpath)
                            r['warm_load'] = True
                        except Exception as e:
                            r['warm_load'] = False
                            r['warm_load_error'] = str(e)
            except Exception:
                # TF not available; mark warm_load as skipped
                for r in results:
                    if r.get('type') == 'keras':
                        r['warm_load'] = None

        # store in cache
        try:
            cache['data'] = results
            cache['ts'] = now
        except Exception:
            pass

        return jsonify({"models": results, "cached": False})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"models": [], "error": str(e)}), 500


@app.route("/api/quant_models", methods=["GET"])
def api_quant_models():
    """List quantized numeric feature models.
    Searches QUANT_MODELS_DIR first, then falls back to IMAGE_MODELS_DIR for backwards compatibility.
    This endpoint inspects joblib files and returns expected_features and optional sidecar input_names.
    Results are cached in-memory using `_quant_models_cache`.
    """
    try:
        # Try the correct location first
        quant_dir = QUANT_MODELS_DIR
        if not os.path.isdir(quant_dir):
            # Fall back to old location for backwards compatibility
            quant_dir = IMAGE_MODELS_DIR
            if not os.path.isdir(quant_dir):
                return jsonify({"models": [], "message": f"Models dir not found: {QUANT_MODELS_DIR} or {IMAGE_MODELS_DIR}"}), 200
        
        # Check cache first
        now = float(__import__('time').time())
        cache = _quant_models_cache
        if cache.get('data') is not None and (now - float(cache.get('ts', 0.0))) < float(cache.get('ttl', 10.0)):
            return jsonify({"models": cache['data'], "cached": True, "ttl": cache['ttl']})

        joblib_files = [f for f in sorted(os.listdir(quant_dir)) if f.lower().endswith('.joblib')]
        out = []
        for jf in joblib_files:
            p = os.path.join(quant_dir, jf)
            expected = None
            input_names = None
            try:
                # attempt lightweight load to inspect attributes
                m = joblib.load(p)
                if hasattr(m, 'n_features_in_'):
                    try:
                        expected = int(getattr(m, 'n_features_in_'))
                    except Exception:
                        expected = None
                elif hasattr(m, 'coef_'):
                    try:
                        expected = int(m.coef_.shape[1])
                    except Exception:
                        expected = None
            except Exception:
                # skip heavy errors; leave expected as None
                expected = None

            # try to read optional sidecar JSON for friendly input names / labels
            try:
                # possible sidecar names: same basename + .json or .joblib.json
                base = os.path.splitext(jf)[0]
                candidates = [os.path.join(quant_dir, base + '.json'), os.path.join(quant_dir, jf + '.json'), os.path.join(quant_dir, base + '.joblib.json')]
                for sc in candidates:
                    if os.path.exists(sc):
                        try:
                            with open(sc, 'r', encoding='utf-8') as sf:
                                data = json.load(sf)
                                # expect {'input_names': [...], 'labels': {...}} optionally
                                if isinstance(data.get('input_names'), list):
                                    input_names = [str(x) for x in data.get('input_names')]
                        except Exception:
                            # ignore malformed sidecar
                            pass
                        break
            except Exception:
                pass

            out.append({"filename": jf, "expected_features": expected, "input_names": input_names})

        # store in cache
        try:
            cache['data'] = out
            cache['ts'] = now
        except Exception:
            # ignore caching errors; still return the list
            pass

        return jsonify({"models": out, "cached": False})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"models": [], "error": str(e)}), 500


@app.route("/api/quant_models/refresh", methods=["POST", "GET"])
def api_quant_models_refresh():
    """Invalidate the in-memory cache for quant model discovery so subsequent calls will re-scan artifacts.
    Accepts POST (preferred) or GET for convenience in manual testing.
    """
    try:
        global _quant_models_cache
        _quant_models_cache['data'] = None
        _quant_models_cache['ts'] = 0.0
        return jsonify({"refreshed": True}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"refreshed": False, "error": str(e)}), 500


@app.route("/api/image_models/refresh", methods=["POST", "GET"])
def api_image_models_refresh():
    """Invalidate the in-memory cache for image model discovery so subsequent calls re-scan artifacts.
    Accepts POST (preferred) or GET for convenience in manual testing.
    """
    try:
        global _image_models_cache
        _image_models_cache['data'] = None
        _image_models_cache['ts'] = 0.0
        return jsonify({"refreshed": True}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"refreshed": False, "error": str(e)}), 500


# ---------------- API: predict_quant (numeric features -> quantized model) ----------------
@app.route("/api/predict_quant", methods=["POST"])
def api_predict_quant():
    """Predict using a quantized (joblib) model from QUANT_MODELS_DIR.
    Expects JSON: {"model": "filename.joblib", "features": [numeric array]}
              OR: {"model_name": "filename.joblib", "features": [numeric array]}
    Returns: {"prediction": label, "confidence": float, "model": filename}
    """
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"error": "invalid_json", "message": "Request body must be valid JSON."}), 400
        
        # Support both 'model' and 'model_name' fields for backwards compatibility
        model_filename = payload.get("model") or payload.get("model_name")
        features = payload.get("features")
        
        if not model_filename:
            return jsonify({"error": "missing_model", "message": "Provide 'model' or 'model_name' field with the joblib filename."}), 400
        if not features or not isinstance(features, list):
            return jsonify({"error": "missing_features", "message": "Provide 'features' as a numeric array."}), 400
        
        # Try QUANT_MODELS_DIR first, then fall back to IMAGE_MODELS_DIR
        model_path = os.path.join(QUANT_MODELS_DIR, model_filename)
        if not os.path.exists(model_path):
            model_path = os.path.join(IMAGE_MODELS_DIR, model_filename)
            if not os.path.exists(model_path):
                return jsonify({"error": "model_not_found", "message": f"Model {model_filename} not found in {QUANT_MODELS_DIR} or {IMAGE_MODELS_DIR}"}), 404
        
        try:
            model = joblib.load(model_path)
        except Exception as e:
            return jsonify({"error": "model_load_failed", "message": str(e)}), 500
        
        # convert features to numpy array and reshape to (1, n_features)
        try:
            X = np.array(features, dtype=float).reshape(1, -1)
        except Exception as e:
            return jsonify({"error": "invalid_features", "message": f"Could not convert features to numeric array: {e}"}), 400
        
        # predict
        try:
            pred_raw = model.predict(X)[0]
        except Exception as e:
            return jsonify({"error": "prediction_failed", "message": str(e)}), 500
        
        # compute confidence
        conf = compute_confidence(model, X)
        
        # try to decode label using label_encoder if available (check for sidecar first)
        pred_label = str(pred_raw)
        model_dir = os.path.dirname(model_path)  # Use the directory where we found the model
        try:
            # check for model-specific sidecar with labels
            base = os.path.splitext(model_filename)[0]
            sidecar_path = os.path.join(model_dir, base + '.json')
            if not os.path.exists(sidecar_path):
                sidecar_path = os.path.join(model_dir, model_filename + '.json')
            
            if os.path.exists(sidecar_path):
                with open(sidecar_path, 'r', encoding='utf-8') as f:
                    sidecar = json.load(f)
                    label_map = sidecar.get('labels')
                    if isinstance(label_map, dict):
                        pred_label = label_map.get(str(pred_raw), str(pred_raw))
                    elif isinstance(label_map, list) and int(pred_raw) < len(label_map):
                        pred_label = label_map[int(pred_raw)]
            else:
                # fallback to global label_encoder from the model directory
                le_path = os.path.join(model_dir, 'label_encoder.joblib')
                if os.path.exists(le_path):
                    le = joblib.load(le_path)
                    pred_label = le.inverse_transform([pred_raw])[0]
        except Exception:
            # if label decoding fails, just use the raw prediction
            pred_label = str(pred_raw)
        
        return jsonify({
            "prediction": pred_label,
            "confidence": conf,
            "model": model_filename
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "server_error", "message": str(e)}), 500


# ---------------- API: predict (uses matching vectorizer) ----------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True, silent=True)
        if not payload or "text" not in payload:
            return jsonify({"error":"No text provided"}), 400
        text = str(payload["text"]).strip()
        model_filename = payload.get("model_filename", None)
        if not model_filename:
            return jsonify({"error":"No model_filename specified"}), 400
        model_path = os.path.join(ART_DIR, model_filename)
        if not os.path.exists(model_path):
            return jsonify({"error":f"Model not found: {model_filename}"}), 404

        # load model
        model = joblib.load(model_path)

        # load vectorizer for this model (strict attempt, but will fallback to tfidf.joblib if present)
        try:
            vectorizer, vec_name = load_vectorizer_for_model(model_filename)
        except FileNotFoundError as e:
            return jsonify({"error":"missing_vectorizer","message":str(e),"available_vectorizers": list_vectorizers()}), 400

        # quick input quality checks (gibberish filter)
        if is_gibberish(text):
            return jsonify({"error": "gibberish", "message": "Please enter valid text. Provide a clear description of plant symptoms (e.g., 'leaves turning yellow with brown spots')."}), 400

        # transform and sanity-check dims
        try:
            X = vectorizer.transform([text])
        except Exception as e:
            return jsonify({"error":f"vectorizer transform error: {e}"}), 500

        # quick check expected vs produced (best-effort)
        produced = X.shape[1] if len(X.shape) >= 2 else None
        expected = None
        if hasattr(model, "n_features_in_"):
            try:
                expected = int(model.n_features_in_)
            except Exception:
                expected = None
        elif hasattr(model, "coef_"):
            try:
                expected = int(model.coef_.shape[1])
            except Exception:
                expected = None

        if expected is not None and produced is not None and expected != produced:
            # attempt to find and apply a saved feature-selector for this model
            selector, sel_name = load_selector_for_model_if_present(model_filename)
            if selector is not None:
                try:
                    X_sel = selector.transform(X)
                    new_produced = X_sel.shape[1] if len(X_sel.shape) >= 2 else None
                    if new_produced == expected:
                        # use X_sel for prediction
                        X = X_sel
                    else:
                        return jsonify({
                            "error": "dimension_mismatch",
                            "message": (
                                f"Model expects {expected} features but vectorizer '{vec_name}' produced {produced} features. "
                                f"A selector '{sel_name}' was found and applied but produced {new_produced} features (still not matching)."
                            ),
                            "model": model_filename,
                            "expected_features": expected,
                            "produced_features": produced,
                            "produced_after_selector": new_produced,
                            "vectorizer_used": vec_name,
                            "selector_used": sel_name
                        }), 400
                except Exception as e:
                    return jsonify({
                        "error": "selector_apply_error",
                        "message": f"Found selector '{sel_name}' but failed to apply it: {e}",
                        "model": model_filename,
                        "vectorizer_used": vec_name,
                        "selector_used": sel_name
                    }), 500
            else:
                # No selector found; provide guidance so the user can fix artifacts
                # list common expected selector filenames to help the user
                tried = [f"{model_filename.split('__')[0]}_selector.joblib",
                         f"{model_filename.split('__')[0]}_kbest.joblib",
                         f"{model_filename.split('__')[0]}_fs.joblib"]
                return jsonify({
                    "error": "dimension_mismatch",
                    "message": f"Model expects {expected} features but vectorizer '{vec_name}' produced {produced} features.",
                    "model": model_filename,
                    "expected_features": expected,
                    "produced_features": produced,
                    "vectorizer_used": vec_name,
                    "advice": (
                        "This model appears to require a feature-selection step (e.g. SelectKBest) that was applied during training but the selector artifact is not present. "
                        "To fix: save the feature-selector (or save the full pipeline Vectorizer->Selector->Estimator) to the artifacts folder using one of the filenames: "
                        + ", ".join(tried)
                    )
                }), 400

        # predict
        pred_raw = model.predict(X)[0]
        try:
            le = get_label_encoder()
            if le is not None:
                pred_label = le.inverse_transform([pred_raw])[0]
            else:
                pred_label = str(pred_raw)
        except Exception:
            pred_label = str(pred_raw)

        # compute confidence (supports predict_proba or decision_function fallback)
        conf = compute_confidence(model, X)

        # low-confidence handling: return a clear response so the UI can show a helpful message
        if conf is not None and conf < CONFIDENCE_THRESHOLD:
            return jsonify({
                "error": "low_confidence",
                "message": "Model confidence is low for this input. Please rephrase or provide more details.",
                "prediction": pred_label,
                "confidence": conf,
                "used_model": model_filename,
                "vectorizer_used": vec_name
            }), 200

        return jsonify({"prediction": pred_label, "confidence": conf, "used_model": model_filename, "vectorizer_used": vec_name})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------- Root UI ----------------
@app.route("/")
def root():
    return render_template("home.html")

@app.route("/text")
def text_prediction():
    return render_template("text.html")

@app.route("/quant")
def quant_prediction():
    return render_template("quant.html")

@app.route("/image")
def image_prediction():
    return render_template("image.html")

# ---------------- Startup ----------------
if __name__ == "__main__":
    if not os.path.isdir(ART_DIR):
        print("Warning: artifacts dir missing:", ART_DIR)
    print("Artifacts dir:", ART_DIR)
    print("Found model files:", safe_list_joblib_files())
    print("Available vectorizers:", list_vectorizers())
    # Run without the debug reloader to avoid intermittent restarts during local testing.
    app.run(host="127.0.0.1", port=DEFAULT_PORT, debug=False, use_reloader=False)
