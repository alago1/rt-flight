import json
import logging
from pprint import pprint

import zmq

# import os
# import django
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_ui.settings')
# django.setup()

from django.contrib.gis.geos import Point
from markers.models import Marker

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
syncclient = context.socket(zmq.REQ)


def configure_zmq(address, port):
    print(f"Subscriber connecting to tcp://{address}:{port}")
    subscriber.connect(f"tcp://{address}:{port}")
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')

    print(f"Syncclient connecting to tcp://{address}:{port+1}")
    syncclient.connect("tcp://localhost:5557")


def receive_messages():
    try:
        synced = False

        print('waiting for server to be ready')
        while True:
            msg = subscriber.recv()

            if msg == b'':
                if not synced:
                    syncclient.send(b'')
                    synced = True
                continue
            
            json_msg = json.loads(msg)
            logging.info(f"Received message: {json_msg}")

            for bbox in json_msg:
                marker = Marker()
                marker.name = 'test name'
                marker.location = Point(bbox['longitude'], bbox['latitude'], srid=4326)  # using srid=4326 for WGS84
                marker.confidence = bbox['confidence']
                marker.radius = bbox['radius']
                marker.save()
            
    except KeyboardInterrupt:
        logging.info("Received interrupt, stopping worker thread")
    
    subscriber.close()
    syncclient.close()
    context.term()
