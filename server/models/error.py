from dataclasses import dataclass


@dataclass(slots=True)
class DetectionError:
    error_msg: str


@dataclass(slots=True)
class HeaderError:
    error_msg: str
