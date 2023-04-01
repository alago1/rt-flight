import smopy
import tkinter as tk
from tkinter import *
from tkinter import ttk
import grpc
import os, sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "protos"))
import messaging_pb2
import messaging_pb2_grpc

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random
import numpy as np
from sklearn.cluster import DBSCAN


channel = grpc.insecure_channel('localhost:50051')
stub = messaging_pb2_grpc.MessagingServiceStub(channel)

root = tk.Tk()
root.geometry("800x1100")

DATASET_TOP_LEFT_GPS = np.array((12.86308254761559, 77.5151947517078))
DATASET_TOP_RIGHT_GPS = np.array((12.863010715187013, 77.52267023737696))
DATASET_BOT_LEFT_GPS = np.array((12.859008245256549, 77.5151541499705))
DATASET_BOT_RIGHT_GPS = np.array((12.858936436333265, 77.52262951527761))
DATASET_CORNER_GPS_COORDS = np.array([DATASET_TOP_LEFT_GPS, DATASET_TOP_RIGHT_GPS, DATASET_BOT_LEFT_GPS, DATASET_BOT_RIGHT_GPS])

# lat_center = 29.643946
# lon_center = -82.355659
lat_center = DATASET_CORNER_GPS_COORDS[:, 0].mean()
lon_center = DATASET_CORNER_GPS_COORDS[:, 1].mean()

# lat_mile = 0.0144927536231884
# lon_mile = 0.0181818181818182
lat_min = DATASET_CORNER_GPS_COORDS[:, 0].min()
lat_max = DATASET_CORNER_GPS_COORDS[:, 0].max()
lon_min = DATASET_CORNER_GPS_COORDS[:, 1].min()
lon_max = DATASET_CORNER_GPS_COORDS[:, 1].max()
# lat_min = lat_center - (15 * lat_mile)
# lat_max = lat_center + (15 * lat_mile)
# lon_min = lon_center - (15 * lon_mile)
# lon_max = lon_center + (15 * lon_mile)

map_ = smopy.Map((lat_min, lon_min, lat_max, lon_max), z=16)
map_img = map_.to_pil()

fig = Figure(frameon=False)
ax = fig.add_subplot(111)
ax.axis("off")

ax.imshow(map_img, interpolation="lanczos", origin="upper")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

table_view = tk.Frame(root)
table_view.pack()

table = ttk.Treeview(table_view)

table["columns"] = ("UUID", "lat", "long", "radius", "conf")

table.column("#0", width=0, stretch=tk.YES)
table.column("UUID", anchor=tk.CENTER, width=100)
table.column("lat", anchor=tk.CENTER, width=150)
table.column("long", anchor=tk.CENTER, width=150)
table.column("radius", anchor=tk.CENTER, width=75)
table.column("conf", anchor=tk.CENTER, width=100)

table.heading("#0", text="", anchor=tk.CENTER)
table.heading("UUID", text="UUID", anchor=tk.CENTER)
table.heading("lat", text="lat", anchor=tk.CENTER)
table.heading("long", text="long", anchor=tk.CENTER)
table.heading("radius", text="radius", anchor=tk.CENTER)
table.heading("conf", text="conf", anchor=tk.CENTER)

table.pack()

uuid_count = 0

def process_response_data(data, eps=0.0005, min_samples=3):
    if not data:
        return [], []

    # each sample is [latitude, longitude, radius, confidence]
    samples = np.array([[float(x) for x in row.split()] for row in data])

    # apply DBSCAN to find clusters
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(samples[:, :2])
    return samples, labels

def update_map():
    global uuid_count
    
    table.delete(*table.get_children())
    ax.clear()
    ax.imshow(map_img, interpolation="lanczos", origin="upper")

    response = stub.RequestProcessedData(messaging_pb2.ProcessedDataRequest(request="Request data"))
    data, cluster_labels = process_response_data(response.processed_data)

    pixel_locations = []

    for uuid_count, data_point in enumerate(data):
        lat, lon, rad, conf = data_point

        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            print(f"Point {lat}, {lon} is out of bounds. Won't be visible.")

        x, y = map_.to_pixels(lat, lon)
        pixel_locations.append((x, y))

        table.insert(
            parent="",
            index="end",
            iid=uuid_count,
            text="",
            values=list(str(v) for v in (uuid_count, lat, lon, rad, conf)),
        )

    pixel_locations = np.array(pixel_locations)
    if len(pixel_locations) > 0:
        ax.scatter(pixel_locations.T[0], pixel_locations.T[1], s=data[:, 2], c=cluster_labels, cmap="Spectral", alpha=0.2)
    
    table.yview_moveto(1)
    canvas.draw()
    root.after(6000, update_map)


update_map()
root.mainloop()
