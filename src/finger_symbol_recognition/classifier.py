from pathlib import Path

import numpy as np
import onnxruntime as ort


class EMNISTClassifier:
    CLASSES = (
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "n",
        "q",
        "r",
        "t",
    )

    MEAN = 0.1751
    STD = 0.3332

    def __init__(self, model_path: str | Path):
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict_top_k(
        self, img_28x28: np.ndarray, k: int = 3
    ) -> list[tuple[str, float]]:
        if img_28x28.ndim != 2 or img_28x28.shape != (28, 28):
            raise ValueError(
                f"Expected image with shape (28, 28), got shape {img_28x28.shape}"
            )

        if img_28x28.dtype == np.uint8:
            tensor = img_28x28.astype(np.float32) / 255.0
        else:
            tensor = img_28x28.astype(np.float32)

        tensor = (tensor - self.MEAN) / self.STD
        tensor = np.expand_dims(tensor, axis=(0, 1))

        raw_outputs = self.session.run(None, {self.input_name: tensor})
        logits = raw_outputs[0][0]

        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        top_indices = np.argsort(probabilities)[::-1][:k]
        return [(self.CLASSES[idx], float(probabilities[idx])) for idx in top_indices]

    def predict(self, img_28x28: np.ndarray) -> tuple[str, float]:
        top_1 = self.predict_top_k(img_28x28, k=1)
        return top_1[0]
