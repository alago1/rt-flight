import grpc
from concurrent import futures
import time
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "protos"))
import messaging_pb2
import messaging_pb2_grpc

class MessagingService(messaging_pb2_grpc.MessagingServiceServicer):
    def __init__(self):
        self.buffer = []

    def SendData(self, request, context):
        self.buffer.append(request.data)
        print(request.data)
        return messaging_pb2.DataResponse(status="Data received and added to buffer")

    def RequestProcessedData(self, request, context):
        if not self.buffer:
            return messaging_pb2.ProcessedDataResponse()
        processed_data = self.buffer
        self.buffer = []
        return messaging_pb2.ProcessedDataResponse(processed_data=processed_data)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    messaging_pb2_grpc.add_MessagingServiceServicer_to_server(MessagingService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
