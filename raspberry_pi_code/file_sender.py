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
    img_path = "output_image.jpg"
    out = stub.GetBoundingBoxes(messaging_pb2.File_Payload(path=img_path))
   
    print(out.bboxes)
if __name__ == '__main__':
    run()
