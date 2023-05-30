import multiprocessing as mp
from typing import Any, Sequence, TypeVar, List

from layers.layer import PipelineLayer

T = TypeVar("T", bound='PipelineLayer')

def run_layer(layer: T, q: mp.Queue, args):
    q.put(layer.run(*args))


class ParallelLayer(PipelineLayer):
    layers: Sequence[T] = []
    queues: Sequence[mp.Queue] = []

    def __init__(self, layers: Sequence[T]):
        self.layers = layers
        self.queues = [mp.Queue() for _ in layers]


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
        for layer, q, args in zip(self.layers, self.queues, layer_args):
            p = mp.Process(target=run_layer, args=(layer, q, args))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
        
        outputs = [q.get() for q in self.queues]
        return outputs
