from dataclasses import dataclass

@dataclass
class DetectionError:
    error_msg: str


@dataclass
class HeaderError:
    error_msg: str
