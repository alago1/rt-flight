import onnxruntime

from engines.engine import AbstractEngine

class OnnxEngine(AbstractEngine):
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.model = self.load_model(**kwargs)
    
    def load_model(self, **kwargs):
        session = onnxruntime.InferenceSession(self.model_path, **kwargs)
        return session

    def get_input_shape(self):
        _, height, width, _ = self.model.get_inputs()[0].shape
        return (height, width)

    def __call__(self, input):
        return self.model.run(None, {"image_input": input})