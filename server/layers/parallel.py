import multiprocessing as mp
from typing import Any, Sequence, TypeVar, List
import logging

from .layer import PipelineLayer

T = TypeVar("T", bound='PipelineLayer')

def run_layer(layer: T, output_queue: mp.Queue, error_queue: mp.Queue, args):
    try:
        output_queue.put(layer.run(*args))
        error_queue.put(None)
    except Exception as e:
        output_queue.put(None)
        error_queue.put(e)


class ParallelLayer(PipelineLayer):
    layers: Sequence[T] = []
    outputs: Sequence[mp.Queue] = []
    errors: Sequence[mp.Queue] = []

    def __init__(self, layers: Sequence[T]):
        self.layers = layers
        self.outputs = [mp.Queue() for _ in layers]
        self.errors = [mp.Queue() for _ in layers]


    def run(self, layer_args: Sequence[Any], share_input: bool = False):
        """
        Run the layers in parallel, passing the arguments in layer_args to each layer.
        For example if two layers have the inputs (a, b) and (c, d) respectively, then
        layer_args should be [(a, b), (c, d)].

        If share_input is True, the layer_args sequence will be passed to each layer.
        In this case, layer_args (a, b) will be passed to each layer
        (i.e. layer1.run(a, b) and layer2.run(a, b)).

        Returns a list of outputs from each layer in the same order as they were passed in.
        """

        if share_input:
            layer_args = [layer_args for _ in self.layers]

        processes: List[mp.Process] = []
        for layer, out, err, args in zip(self.layers, self.outputs, self.errors, layer_args):
            p = mp.Process(target=run_layer, args=(layer, out, err, args))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        outputs = [q.get() for q in self.outputs]
        errors = [e.get() for e in self.errors]

        if any(errors):
            logging.error(f"Errors occurred while running parallel layers: {errors}")
            raise [e for e in errors if e is not None][0]  # raises the first error, logs all of them

        return outputs
