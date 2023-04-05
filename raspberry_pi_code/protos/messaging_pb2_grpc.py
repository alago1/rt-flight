# pylint: disable=too-few-public-methods
# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import messaging_pb2 as messaging__pb2


class MessagingServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetBoundingBoxes = channel.unary_unary(
                '/messaging.MessagingService/GetBoundingBoxes',
                request_serializer=messaging__pb2.Filepath.SerializeToString,
                response_deserializer=messaging__pb2.BBoxes.FromString,
                )


class MessagingServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetBoundingBoxes(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MessagingServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetBoundingBoxes': grpc.unary_unary_rpc_method_handler(
                    servicer.GetBoundingBoxes,
                    request_deserializer=messaging__pb2.Filepath.FromString,
                    response_serializer=messaging__pb2.BBoxes.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'messaging.MessagingService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MessagingService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetBoundingBoxes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/messaging.MessagingService/GetBoundingBoxes',
            messaging__pb2.Filepath.SerializeToString,
            messaging__pb2.BBoxes.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
