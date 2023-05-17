import zmq
from bbox import BBox
import pickle
import io
from pprint import pprint

context = zmq.Context()

#  Socket to talk to server
print("Connecting to Model Network")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

img_path = "../data/dota_demo.jpg"

socket.send_string(img_path)
message = socket.recv()

arr = pickle.load(io.BytesIO(message))

pprint(arr)
