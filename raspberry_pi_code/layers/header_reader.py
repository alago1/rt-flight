from .layer import PipelineLayer
from models.header_metadata import HeaderMetadata
from util.logger import log_time

class HeaderReader(PipelineLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @log_time
    def run(self, image_path: str):
        metadata = HeaderMetadata.read(image_path, self._logger)
        self._logger.info(f"Read metadata: {metadata}")
        return metadata
