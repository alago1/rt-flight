from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class BBoxes(_message.Message):
    __slots__ = ["jsondata"]
    JSONDATA_FIELD_NUMBER: _ClassVar[int]
    jsondata: str
    def __init__(self, jsondata: _Optional[str] = ...) -> None: ...

class File_Payload(_message.Message):
    __slots__ = ["jsondata", "path"]
    JSONDATA_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    jsondata: str
    path: str
    def __init__(self, path: _Optional[str] = ..., jsondata: _Optional[str] = ...) -> None: ...
