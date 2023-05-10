import grpc
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "protos"))
import messaging_pb2
import messaging_pb2_grpc

def run(port_num):
    channel = grpc.insecure_channel(f'localhost:{port_num}')
    stub = messaging_pb2_grpc.MessagingServiceStub(channel)
    img_path = "2023-04-09_19-58-14_RGB.jpg" # has GPSDirection

    out = stub.GetBoundingBoxes(messaging_pb2.File_Payload(path=img_path))
    print(out.bboxes)
if __name__ == '__main__':
    port_arg = sys.argv[1] if len(sys.argv) > 1 else 50951
    run(port_num=port_arg)
