
import numpy as np
import pycuda.autoinit  # required to initialize cuda
import pycuda.driver as cuda
import tensorrt as trt

from .engine import AbstractEngine

# backwards compatibility for old api calls on the jetson
setattr(np, 'bool', bool)
setattr(np, 'int', int)


class TensorrtEngine(AbstractEngine):
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.cuda_ctx = cuda.Device(0).make_context()
        self.model = self.load_model(**kwargs)
    
    def load_model(self, **kwargs):
        model = TrtYOLOv3(self.model_path, cuda_ctx=self.cuda_ctx, **kwargs)
        return model

    def get_input_shape(self):
        hw, c = self.model.input_shape
        return (hw, hw)

    def __call__(self, input):
        return self.model.detect(input)
    
    def __del__(self):
        self.cuda_ctx.pop()
        del self.model


# This code is adapted from these Yolov3 TRT examples:
# https://github.com/NVIDIA/TensorRT/blob/main/samples/python/yolov3_onnx/onnx_to_tensorrt.py
# https://github.com/jkjung-avt/tensorrt_demos/blob/master/utils/yolo_with_plugins.py
# https://github.com/yqlbu/TRT-yolov3/blob/master/utils/yolov3.py

class HostDeviceMem():
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.device
        del self.host


def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    output_idx = 0
    stream = cuda.Stream()
    for binding in engine:
        binding_dims = engine.get_binding_shape(binding)
        if len(binding_dims) == 4:
            # explicit batch case (TensorRT 7+)
            size = trt.volume(binding_dims)
        elif len(binding_dims) == 3:
            # implicit batch case (TensorRT 6 or older)
            size = trt.volume(binding_dims) * engine.max_batch_size
        else:
            raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_idx += 1
    assert len(inputs) == 1
    assert len(outputs) > 0
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """do_inference (for TensorRT 6.x or lower)

    This function is generalized for multiple inputs/outputs.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size,
                          bindings=bindings,
                          stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def do_inference_v2(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def get_input_shape(engine):
    """Get input shape of the TensorRT YOLO engine."""
    binding = engine[0]
    assert engine.binding_is_input(binding)
    binding_dims = engine.get_binding_shape(binding)
    if len(binding_dims) == 4:
        return tuple(binding_dims[2:])
    elif len(binding_dims) == 3:
        return tuple(binding_dims[1:])
    else:
        raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))
    

class TrtYOLOv3():
    def _load_engine(self):
        with open(self.model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, model_path, output_shape=None, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and context."""
        self.model_path = model_path
        self.cuda_ctx = cuda_ctx

        if self.cuda_ctx is not None:
            self.cuda_ctx.push()

        self.inference_fn = do_inference if trt.__version__[0] < '7' \
                                         else do_inference_v2
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()
        self.input_shape = get_input_shape(self.engine)

        if output_shape is None:
            output_shape = [(1, 75, 13, 13), (1, 75, 26, 26), (1, 75, 52, 52)]
        
        self.output_shapes = sorted(output_shape, key=lambda x: np.prod(x))

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError("Failed to allocate cuda resources") from e
        finally:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()
        

    def __del__(self):
        """Free CUDA memories."""
        mem_attrs = ['stream', 'outputs', 'inputs']
        for mem_attr in mem_attrs:
            mem_ptr = getattr(self, mem_attr, None)
            
            if mem_ptr is not None:
                del mem_ptr


    def detect(self, img):
        """Detect objects in the input image."""

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(img)

        if self.cuda_ctx is not None:
            self.cuda_ctx.push()

        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)

        if self.cuda_ctx is not None:
            self.cuda_ctx.pop()

        # Before doing post-processing, we need to reshape the outputs
        # as do_inference() will give us flat arrays.
        order = np.argsort(np.prod([o.shape for o in trt_outputs], axis=1))
        trt_outputs = [trt_outputs[i] for i in order]

        trt_outputs = [output.reshape(shape) for output, shape
                       in zip(trt_outputs, self.output_shapes)]
        
        # move channels to last axis (NCHW -> NHWC)
        trt_outputs = [out.transpose((0, 2, 3, 1)) for out in trt_outputs]

        return trt_outputs