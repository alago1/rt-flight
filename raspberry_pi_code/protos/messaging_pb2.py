# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: messaging.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fmessaging.proto\x12\tmessaging\".\n\x0c\x46ile_Payload\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x10\n\x08jsondata\x18\x02 \x01(\t\"\x1a\n\x06\x42\x42oxes\x12\x10\n\x08jsondata\x18\x01 \x01(\t2T\n\x10MessagingService\x12@\n\x10GetBoundingBoxes\x12\x17.messaging.File_Payload\x1a\x11.messaging.BBoxes\"\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'messaging_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _FILE_PAYLOAD._serialized_start=30
  _FILE_PAYLOAD._serialized_end=76
  _BBOXES._serialized_start=78
  _BBOXES._serialized_end=104
  _MESSAGINGSERVICE._serialized_start=106
  _MESSAGINGSERVICE._serialized_end=190
# @@protoc_insertion_point(module_scope)
