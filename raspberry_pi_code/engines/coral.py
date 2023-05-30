import numpy as np
from pycoral.adapters import common
from pycoral.utils import edgetpu

from engines.engine import AbstractEngine


class CoralEngine(AbstractEngine):
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.model = self.load_model(**kwargs)

    def load_model(self, **kwargs):
        model = edgetpu.make_interpreter(self.model_path, **kwargs)
        model.allocate_tensors()
        return model

    def get_input_shape(self):
        return common.input_size(self.model)[::-1]  # (height, width)

    def __call__(self, input):
        common.set_input(self.model, input)
        self.model.invoke()
        output_details = self.model.get_output_details()

        outputs = []
        for i in range(len(output_details)):
            out = self.model.get_tensor(output_details[i]['index'])
            scale, zero_point = dict.get(output_details[i], 'quantization', (1, 0))
            out = (out.astype(np.float32) - zero_point) * scale

            outputs.append(out)
        
        return outputs
