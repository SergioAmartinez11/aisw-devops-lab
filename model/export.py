import numpy as np
import onnx
import onnxruntime as ort
import torch
from cnn import MiniCNN

MODEL_PATH = "../artifacts/model.pth"
ONNX_PATH  = "../artifacts/model.onnx"


def export():
    model = MiniCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,   # fusiona operaciones constantes → más rápido en edge
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={              # batch size dinámico → flexible en producción
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    model_proto = onnx.load(ONNX_PATH)
    onnx.save(model_proto, ONNX_PATH, save_as_external_data=False)
    print(f"✓ ONNX model exported → {ONNX_PATH}")


def verify():
    # 1. Validación estructural del grafo ONNX
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print("✓ ONNX graph is valid")

    # 2. Inferencia real con ONNX Runtime
    session = ort.InferenceSession(ONNX_PATH)
    dummy   = np.random.randn(1, 1, 28, 28).astype(np.float32)
    output  = session.run(["output"], {"input": dummy})

    print(f"✓ Inference OK | output shape: {output[0].shape} | predicted class: {output[0].argmax()}")


if __name__ == "__main__":
    export()
    verify()