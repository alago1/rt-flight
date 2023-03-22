import grpc
import os, sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "protos"))
import messaging_pb2
import messaging_pb2_grpc
import time

def GetServerResponse():
    lat_center = 29.643946
    lon_center = -82.355659

    lat_mile = 0.0144927536231884
    lon_mile = 0.0181818181818182
    lat_min = lat_center - (15 * lat_mile)
    lat_max = lat_center + (15 * lat_mile)
    lon_min = lon_center - (15 * lon_mile)
    lon_max = lon_center + (15 * lon_mile)

    lat, lon = generateCoords(lon_min, lon_max, lat_min, lat_max)
    rad = random.uniform(0, 2000)
    conf = random.uniform(0, 100)

    return f"{lat} {lon} {rad} {conf}"

def generateCoords(lon_min, lon_max, lat_min, lat_max):
    lat = random.uniform(lat_min, lat_max)
    lon = random.uniform(lon_min, lon_max)
    return lat, lon

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = messaging_pb2_grpc.MessagingServiceStub(channel)

    while True:
        data = GetServerResponse()
        response = stub.SendData(messaging_pb2.DataRequest(data=data))
        print(response.status)
        time.sleep(2)

if __name__ == '__main__':
    run()
