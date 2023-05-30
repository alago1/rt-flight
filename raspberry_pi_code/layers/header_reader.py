import logging

from .layer import PipelineLayer
from models.header_metadata import HeaderMetadata
from util.logging import log_time

class HeaderReader(PipelineLayer):
    @log_time
    def run(self, image_path: str):
        metadata = HeaderMetadata.read(image_path)
        logging.info(f"Read metadata: {metadata}")
        return metadata
