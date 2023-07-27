import multiprocessing as mp
from typing import Any, Sequence, TypeVar, List, Union
import logging
import threading

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

    def __init__(self, layers: Sequence[T], thread_first=False):
        """
        layers: Sequence of layers to be executed in parallel
        thread_first: If true, will execute the first layer on a thread
        """

        self.layers = layers
        self.outputs = [mp.Queue() for _ in layers]
        self.errors = [mp.Queue() for _ in layers]
        self.thread_first = thread_first


    def run(self, layer_args: Sequence[Any], share_input: bool = False):
        """
        Run the layers in parallel, passing the arguments in layer_args to each layer.
        For example if two layers have the inputs (a, b) and (c, d) respectively, then
        layer_args should be [(a, b), (c, d)].

        If share_input is True, the layer_args sequence will be passed to each layer.
        In this case, layer_args (a, b) will be passed to each layer
        (i.e. layer1.run(a, b) and layer2.run(a, b)).

        By default, the first layer is run in a thread

        Returns a list of outputs from each layer in the same order as they were passed in.
        """

        if share_input or len(self.layers) == 1:
            layer_args = [layer_args for _ in self.layers]

        processes: List[Union[mp.Process, threading.Thread]] = []

        layers, out_queues, error_queues, args = self.layers, self.outputs, self.errors, layer_args

        # if using threads first, run the first layer on a thread
        if self.thread_first:
            t = threading.Thread(target=run_layer, args=(layers[0], out_queues[0], error_queues[0], args[0]))
            t.start()
            processes.append(t)

            layers, out_queues, error_queues, args = layers[1:], out_queues[1:], error_queues[1:], args[1:]

        for layer, out, err, args in zip(layers, out_queues, error_queues, args):
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
