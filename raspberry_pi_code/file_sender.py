import io
import pickle
from pprint import pprint

import zmq

import models.error as error
from models.bbox import BBox

context = zmq.Context()

#  Socket to talk to server
print("Connecting to Model Network")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

img_path = "../data/dota_demo.jpg"

socket.send_string(img_path)
message = socket.recv()

try:
    result = pickle.load(io.BytesIO(message))

    if isinstance(result, list):
        # result is a list of BBox objects
        pprint(result)
    elif isinstance(result, error.DetectionError):
        print(f"Received Detection Error: {result.error_msg}")
    elif isinstance(result, error.HeaderError):
        print(f"Received Header Error: {result.error_msg}")
    else:
        print("Received unknown error")
except pickle.UnpicklingError:
    print("Could not unpickle message")
except Exception as e:
    print("Received unknown error while trying to parse message:")
    print(e)
