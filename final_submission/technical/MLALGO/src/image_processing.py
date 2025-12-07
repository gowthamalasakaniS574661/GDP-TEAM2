from PIL import Image
import numpy as np
import io


def load_image_file(file_storage, target_size=(224, 224)):
    """Load an uploaded file (Flask FileStorage) or bytes-like into an RGB numpy array.
    Returns a float32 array shaped (1, H, W, C) with values in [0,1].
    """
    # accept either a FileStorage or bytes/stream
    try:
        # Flask's FileStorage has .stream attribute
        stream = getattr(file_storage, 'stream', None) or file_storage
        img = Image.open(stream)
    except Exception:
        # fallback: try to read bytes
        if hasattr(file_storage, 'read'):
            data = file_storage.read()
            img = Image.open(io.BytesIO(data))
        else:
            raise

    img = img.convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img).astype('float32') / 255.0
    # return as (1, H, W, C)
    return np.expand_dims(arr, axis=0)


def to_nchw(batch_hw3):
    """Convert (N,H,W,3) to (N,3,H,W)"""
    return np.transpose(batch_hw3, (0, 3, 1, 2))
