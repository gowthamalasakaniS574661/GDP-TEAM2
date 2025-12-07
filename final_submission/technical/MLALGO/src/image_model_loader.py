import os
import json
# numpy is imported lazily inside methods to keep module import lightweight

SUPPORTED_EXT = ('.onnx', '.tflite', '.pt', '.pth')


class QuantModelAdapter:
    def __init__(self, path):
        self.path = path
        self.backend = None
        self.model = None
        self._probe_backend()

    def _probe_backend(self):
        p = self.path.lower()
        if p.endswith('.onnx'):
            self.backend = 'onnx'
        elif p.endswith('.tflite'):
            self.backend = 'tflite'
        elif p.endswith('.pt') or p.endswith('.pth'):
            self.backend = 'torch'
        else:
            self.backend = 'unknown'

    def load(self):
        if self.model is not None:
            return
        if self.backend == 'onnx':
            try:
                import onnxruntime as ort
            except Exception as e:
                raise RuntimeError('onnxruntime not installed') from e
            # prefer CPU provider; allow environment to override
            self.model = ort.InferenceSession(self.path, providers=['CPUExecutionProvider'])
            self._ort = ort
        elif self.backend == 'tflite':
            # try tflite-runtime first, fallback to tensorflow
            try:
                import tflite_runtime.interpreter as tflite
                Interpreter = tflite.Interpreter
            except Exception:
                try:
                    from tensorflow.lite import Interpreter
                    # Alias
                except Exception as e:
                    raise RuntimeError('tflite runtime not available') from e
            # create interpreter
            try:
                self.model = Interpreter(model_path=self.path)
                self.model.allocate_tensors()
            except Exception as e:
                raise RuntimeError(f'failed to load tflite model: {e}') from e
        elif self.backend == 'torch':
            try:
                import torch
            except Exception as e:
                raise RuntimeError('torch not installed') from e
            self.torch = torch
            # try to load jit first for .pt, otherwise load state dict
            try:
                if self.path.endswith('.pt'):
                    self.model = torch.jit.load(self.path, map_location='cpu')
                else:
                    self.model = torch.load(self.path, map_location='cpu')
                self.model.eval()
            except Exception as e:
                raise RuntimeError(f'failed to load torch model: {e}') from e
        else:
            raise RuntimeError('unsupported model format')

    def predict(self, image_np):
        """image_np expected shape (1,H,W,3), float32 in [0,1]
        Returns numpy array of model outputs.
        """
        import numpy as np
        self.load()
        if self.backend == 'onnx':
            # ONNX runtimes often expect NCHW
            ort_inp = image_np
            if image_np.ndim == 4 and image_np.shape[-1] == 3:
                ort_inp = np.transpose(image_np, (0, 3, 1, 2)).astype(np.float32)
            else:
                ort_inp = image_np.astype(np.float32)
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: ort_inp})
            return np.array(outputs[0])
        elif self.backend == 'tflite':
            inp_details = self.model.get_input_details()
            out_details = self.model.get_output_details()
            input_index = inp_details[0]['index']
            arr = image_np
            # many tflite models expect NHWC
            arr = arr.astype(inp_details[0]['dtype'])
            try:
                self.model.set_tensor(input_index, arr)
            except Exception:
                # try to squeeze channel order
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    arr2 = np.transpose(arr, (0, 3, 1, 2))
                    self.model.set_tensor(input_index, arr2.astype(inp_details[0]['dtype']))
                else:
                    raise
            self.model.invoke()
            out = self.model.get_tensor(out_details[0]['index'])
            return np.array(out)
        elif self.backend == 'torch':
            t = self.torch.from_numpy(np.transpose(image_np, (0, 3, 1, 2))).float()
            with self.torch.no_grad():
                out = self.model(t)
            if isinstance(out, (list, tuple)):
                out = out[0]
            return out.cpu().numpy()
        else:
            raise RuntimeError('unsupported backend')


def discover_models(directory):
    """Scan directory for supported quantized model files and return a dict name->adapter
    Name is the basename without extension.
    """
    models = {}
    if not os.path.isdir(directory):
        return models
    for fname in os.listdir(directory):
        if not any(fname.lower().endswith(ext) for ext in SUPPORTED_EXT):
            continue
        path = os.path.join(directory, fname)
        name = os.path.splitext(fname)[0]
        models[name] = QuantModelAdapter(path)
    return models
