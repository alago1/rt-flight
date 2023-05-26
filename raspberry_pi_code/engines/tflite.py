import tflite_runtime.interpreter as tflite

from engines.engine import AbstractEngine

class TfliteEngine(AbstractEngine):
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.model = self.load_model(**kwargs)
    
    def load_model(self, **kwargs):
        model = tflite.Interpreter(self.model_path, num_threads=4, **kwargs)
        model.allocate_tensors()
        return model

    def get_input_shape(self):
        _, height, width, _ = self.model.get_input_details()[0]["shape"]
        return (height, width)

    def __call__(self, input):
        tensor_index = self.net.get_input_details()[0]["index"]
        self.model.set_tensor(tensor_index, input)
        self.model.invoke()
        output_details = self.model.get_output_details()
        output = [self.model.get_tensor(output_details[i]["index"]) for i in range(len(output_details))]

        return output
