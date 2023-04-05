import grpc
import os, sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "protos"))
import messaging_pb2
import messaging_pb2_grpc
import time

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = messaging_pb2_grpc.MessagingServiceStub(channel)
    
    stub.GetBoundingBoxes(messaging_pb2.Filepath(path="SomePath/This/Here/Now"))

if __name__ == '__main__':
    run()
