import traceback
import sys
import os
import io
import pickle
from pprint import pprint
from pathlib import Path
import time

import zmq
import numpy as np

from util.zmq_util import message_queue, start_server, server_join

project_path = Path(__file__).parent.parent
sys.path.append(str(project_path))

from server.models.error import DetectionError, HeaderError
from server.models.bbox import BBox, BBoxes


context = zmq.Context()

#  Socket to talk to server
print("Connecting to Model Network")
model_socket = context.socket(zmq.REQ)
model_socket.connect("tcp://localhost:5555")

PUBLISHER_PORT = 5556
SYNC_PORT = 5557

start_server(context, min_subs=1, publisher_port=PUBLISHER_PORT, sync_port=SYNC_PORT)


images_folder = project_path / "data_ignore" / "sequential_with_exif"
images = [images_folder / img for img in os.listdir(str(images_folder)) if img.lower().endswith(".jpg")]
images.sort(key=lambda x: int(str(x)[-8:-4]))  # hack to sort by image number

time_metrics = []

for i, img_path in enumerate(images):
    print(f"Sending image {i + 1} of {len(images)}: {img_path}")
    s = time.perf_counter()
    model_socket.send_string(str(img_path))
    message = model_socket.recv()
    e = time.perf_counter()
    time_metrics.append(e-s)
    print(f"Image {i+1} end-to-end time: {1000*(e-s):.1f}ms")

    try:
        result = pickle.load(io.BytesIO(message))

        if isinstance(result, BBoxes):
            message_queue.put(result)
            pprint(result)
        elif isinstance(result, DetectionError):
            print(f"Received Detection Error: {result.error_msg}")
        elif isinstance(result, HeaderError):
            print(f"Received Header Error: {result.error_msg}")
        else:
            print("Received unknown error")
            print(result)
    except pickle.UnpicklingError:
        print("Could not unpickle message")
        sys.exit(1)
    except Exception as e:
        print("Received unknown error while trying to parse message:")
        print(traceback.format_exc())
        sys.exit(1)

avg_time = np.mean(time_metrics)
std_dev = np.std(time_metrics)
print(f"Time average: {1000*avg_time}ms")
print(f"Standard deviation: {1000*std_dev}ms")


server_join()
model_socket.close()
context.term()