# pylint: disable=missing-function-docstring
import grpc
from concurrent import futures
import os, sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "mavlink_sim"))

import mavlink_pb2_grpc as pb2_grpc
import mavlink_pb2 as pb2


def generateCoords(lon_min, lon_max, lat_min, lat_max):
    lat = random.uniform(lat_min, lat_max)
    lon = random.uniform(lon_min, lon_max)
    return lat, lon


class UnaryService(pb2_grpc.MavlinkServicer):
    def __init__(self, *args, **kwargs):
        pass

    def GetServerResponse(self, request, context):
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

        result = f"{lat} {lon} {rad} {conf}"
        result = {"message": result, "received": True}

        return pb2.MessageResponse(**result)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_MavlinkServicer_to_server(UnaryService(), server)
    server.add_insecure_port("[::]:50052")
    server.start()
    server.wait_for_termination()

serve()
