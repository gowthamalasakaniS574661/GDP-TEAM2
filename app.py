# app.py  -- Option 1: only expose compatible models (matching *_tfidf.joblib)
from flask import Flask, request, jsonify, render_template
import os, csv, joblib, traceback
import numpy as np

app = Flask(__name__)

ART_DIR = os.path.join("MLALGO", "text_model_artifacts")
LABEL_PATH = os.path.join(ART_DIR, "label_encoder.joblib")
SUMMARY_CSV = os.path.join(ART_DIR, "training_summary_text_models.csv")
DEFAULT_PORT = 5001

# prediction behavior tuning (configurable via env var)
CONFIDENCE_THRESHOLD = float(os.getenv('CONF_THRESHOLD', '0.60'))  # below this value we return a low-confidence response

import itertools

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

# try load label encoder (optional)
_label_encoder = None
try:
    if os.path.exists(LABEL_PATH):
        _label_encoder = joblib.load(LABEL_PATH)
        print("Label encoder loaded.")
except Exception as e:
    print("Could not load label encoder:", e)

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
        return jsonify({"models": compatible_models, "available_vectorizers": available_vecs})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"models": [], "available_vectorizers": list_vectorizers(), "error": str(e)}), 500

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
        if vowel_ratio_all < 0.12:
            return True
    words = s.split()
    # require at least two short words to be considered a valid symptom phrase
    if len(words) < 2:
        # allow two-character abbreviations only if they contain vowels
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
        if vowel_ratio < 0.18:
            return True
    # detect long consonant runs inside any token (e.g. 'JHCHGKXBKJSAXHJ')
    for w in words:
        runs = [len(list(g)) for _, g in itertools.groupby(w)]
        # consonant runs (no vowels) longer than 6 are suspicious
        max_run = 0
        cur_run = 0
        for ch in w.lower():
            if ch.isalpha() and ch not in 'aeiou':
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 0
        if max_run >= 6:
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
            return jsonify({"error": "gibberish", "message": "Input looks like gibberish. Please provide a clear description of symptoms (e.g. leaves turning yellow with spots)."}), 200

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
            if _label_encoder is not None:
                pred_label = _label_encoder.inverse_transform([pred_raw])[0]
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
    return render_template("index.html")

# ---------------- Startup ----------------
if __name__ == "__main__":
    if not os.path.isdir(ART_DIR):
        print("Warning: artifacts dir missing:", ART_DIR)
    print("Artifacts dir:", ART_DIR)
    print("Found model files:", safe_list_joblib_files())
    print("Available vectorizers:", list_vectorizers())
    app.run(host="127.0.0.1", port=DEFAULT_PORT, debug=True)
