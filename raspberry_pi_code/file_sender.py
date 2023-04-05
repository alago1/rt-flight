import grpc
import os, sys
import random
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "protos"))
import messaging_pb2
import messaging_pb2_grpc
import time

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = messaging_pb2_grpc.MessagingServiceStub(channel)
    
    img_path =  "/Users/sahajpatel/Downloads/Snr_Proj/code/rt-flight/realtime_ui/data/IMG_0064.jpg"
    data = {
        "longitude": -82.89811490277778,
        "latitude": 30.30491903277778,
        "altitude": 96.91,
        "heading": 318.5,
        "sensorw": 6.17,
        "sensorh": 4.55,
        "focal_length": 7.2,
        "imagew": 5184,
        "imageh": 3888
    }
    json_data = json.dumps(data)
    
    stub.GetBoundingBoxes(messaging_pb2.File_Payload(path=img_path,
                                                     jsondata=json_data))

if __name__ == '__main__':
    run()
