import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

class MedicalClassifierPipeline:
    def __init__(self, model_dir: str = "model"):
        self.model_path = os.path.join(model_dir, "model_quantized.onnx")
        self.tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        
        # Load ONNX session optimized for low RAM
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(self.model_path, options, providers=["CPUExecutionProvider"])
        
        # Load Tokenizer
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(length=128, pad_id=0, pad_token="[PAD]")
        
        self.categories = ["cardiovascular", "hepatorenal", "oncológico", "neurológico"]

    def predict(self, titulo: str, resumen: str):
        text = f"{titulo} [SEP] {resumen}"
        encoded = self.tokenizer.encode(text)
        
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        outputs = self.session.run(None, inputs)
        logits = outputs[0][0]
        
        # Sigmoid activation for multi-label classification
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        probs = sigmoid(logits)
        
        pred_probs = {self.categories[i]: float(probs[i]) for i in range(len(self.categories))}
        selected_categories = [cat for cat, prob in pred_probs.items() if prob >= 0.5]
        
        if not selected_categories:
            best_cat = max(pred_probs, key=pred_probs.get)
            selected_categories = [best_cat]
            
        return {
            "categorias": selected_categories,
            "probabilidades": pred_probs
        }
