from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BBox(_message.Message):
    __slots__ = ["confidence", "latitude", "longitude", "radius"]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    LATITUDE_FIELD_NUMBER: _ClassVar[int]
    LONGITUDE_FIELD_NUMBER: _ClassVar[int]
    RADIUS_FIELD_NUMBER: _ClassVar[int]
    confidence: float
    latitude: float
    longitude: float
    radius: float
    def __init__(self, latitude: _Optional[float] = ..., longitude: _Optional[float] = ..., radius: _Optional[float] = ..., confidence: _Optional[float] = ...) -> None: ...

class BBoxes(_message.Message):
    __slots__ = ["bboxes"]
    BBOXES_FIELD_NUMBER: _ClassVar[int]
    bboxes: _containers.RepeatedCompositeFieldContainer[BBox]
    def __init__(self, bboxes: _Optional[_Iterable[_Union[BBox, _Mapping]]] = ...) -> None: ...

class File_Payload(_message.Message):
    __slots__ = ["path"]
    PATH_FIELD_NUMBER: _ClassVar[int]
    path: str
    def __init__(self, path: _Optional[str] = ...) -> None: ...
