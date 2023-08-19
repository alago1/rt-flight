import tflite_runtime.interpreter as tflite
import numpy as np

from .engine import AbstractEngine


class TfliteEngine(AbstractEngine):
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.model = self.load_model(**kwargs)
    
    def load_model(self, **kwargs):
        model = tflite.Interpreter(self.model_path, **kwargs)
        model.allocate_tensors()
        return model

    def get_input_shape(self):
        _, height, width, _ = self.model.get_input_details()[0]["shape"]
        return (height, width)

    def __call__(self, input):        
        if np.issubdtype(input.dtype, np.floating):
            input = 255 * input
        
        if input.dtype != np.uint8:
            input = input.astype(np.uint8)

        tensor_index = self.model.get_input_details()[0]["index"]
        self.model.set_tensor(tensor_index, input)
        self.model.invoke()
        output_details = self.model.get_output_details()

        outputs = []
        for i in range(len(output_details)):
            out = self.model.get_tensor(output_details[i]['index'])
            scale, zero_point = dict.get(output_details[i], 'quantization', (1, 0))
            out = (out.astype(np.float32) - zero_point) * scale

            outputs.append(out)

        return outputs
