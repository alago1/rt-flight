import os
import threading

from django.apps import AppConfig
from django.conf import settings



class MarkersConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "markers"

    def ready(self):
        from .zmq_utils import configure_zmq, receive_messages

        configure_zmq(settings.ZMQ_ADDRESS, settings.ZMQ_PORT)
        worker_thread = threading.Thread(target=receive_messages)
        if os.environ.get("RUN_MAIN", None) == "true":
            print("Starting zmq worker thread")
            worker_thread.start()
