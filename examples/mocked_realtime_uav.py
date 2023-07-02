import traceback
import sys
import io
import pickle
from pprint import pprint
from pathlib import Path

import zmq
import numpy as np

from util.zmq_util import message_queue, start_server, server_join
from util.mock_flight_path import mock_flight_path

project_path = Path(__file__).parent.parent
sys.path.append(str(project_path))

from server.models.error import DetectionError, HeaderError
from server.models.bbox import BBox


context = zmq.Context()

#  Socket to talk to server
print("Connecting to Model Network")
model_socket = context.socket(zmq.REQ)
model_socket.connect("tcp://localhost:5555")

PUBLISHER_PORT = 5556
SYNC_PORT = 5557

start_server(context, min_subs=1, publisher_port=PUBLISHER_PORT, sync_port=SYNC_PORT)


DATASET_CORNER_GPS_COORDS = np.array([
    (12.86308254761559, 77.5151947517078),  # top left
    (12.863010715187013, 77.52267023737696),  # top right
    (12.858936436333265, 77.52262951527761),  # bottom right
    (12.859008245256549, 77.5151541499705)  # bottom left
])


for img_path in mock_flight_path(project_path / "data/Blore_Clean.tif", 5, DATASET_CORNER_GPS_COORDS, 100.5, seed=42):
    
    model_socket.send_string(img_path)
    message = model_socket.recv()

    try:
        result = pickle.load(io.BytesIO(message))
        message_queue.put(result)

        if isinstance(result, list):
            # result is a list of BBox objects
            pprint(result)
        elif isinstance(result, DetectionError):
            print(f"Received Detection Error: {result.error_msg}")
        elif isinstance(result, HeaderError):
            print(f"Received Header Error: {result.error_msg}")
        else:
            print("Received unknown error")
    except pickle.UnpicklingError:
        print("Could not unpickle message")
        sys.exit(1)
    except Exception as e:
        print("Received unknown error while trying to parse message:")
        print(traceback.format_exc())
        sys.exit(1)


server_join()
model_socket.close()
context.term()
