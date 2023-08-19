import time
from queue import SimpleQueue
from dataclasses import asdict
import threading

import zmq

message_queue = SimpleQueue()

connected_subs = 0
NUM_SUBSCRIBERS = 1

threads = []

def create_publisher(port, context, min_subscribers=1):
    global NUM_SUBSCRIBERS
    NUM_SUBSCRIBERS = min_subscribers

    def publisher_worker():
        try:
            ui_publisher = context.socket(zmq.PUB)
            ui_publisher.sndhwm = 1100000  # set high water mark so messages aren't dropped
            ui_publisher.bind(f"tcp://*:{port}")

            while connected_subs < NUM_SUBSCRIBERS:
                ui_publisher.send(b'')
                time.sleep(0.1)
            
            while True:
                # blocks and sends message as soon as it's available
                results = message_queue.get()
                ui_publisher.send_json(asdict(results))
        except KeyboardInterrupt:
            pass
        
        ui_publisher.close()
    
    return publisher_worker


def create_syncservice(port, context):
    def syncservice_worker():
        global connected_subs

        syncservice = context.socket(zmq.REP)
        syncservice.bind(f"tcp://*:{port}")

        try:
            while True:
                syncservice.recv()
                connected_subs += 1
                print(f"Connected {connected_subs}/{NUM_SUBSCRIBERS} subscriber(s)")
                syncservice.send(b'')
        except KeyboardInterrupt:
            pass

        syncservice.close()
    
    return syncservice_worker


def start_server(context, min_subs=1, publisher_port=None, sync_port=None):
    pub_port = publisher_port or 5556
    publisher = create_publisher(pub_port, context, min_subscribers=min_subs)
    sync_port = sync_port or pub_port + 1
    sync = create_syncservice(sync_port, context)

    for worker in (publisher, sync):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

def server_join():
    for t in threads:
        t.join()
