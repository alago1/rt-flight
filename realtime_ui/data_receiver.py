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

lat_center = 29.643946
lon_center = -82.355659

lat_mile = 0.0144927536231884
lon_mile = 0.0181818181818182
lat_min = lat_center - (15 * lat_mile)
lat_max = lat_center + (15 * lat_mile)
lon_min = lon_center - (15 * lon_mile)
lon_max = lon_center + (15 * lon_mile)

map_ = smopy.Map((lat_min, lon_min, lat_max, lon_max), z=10)
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


def update_map():
    global uuid_count
    global client
    global scatter
    
    table.delete(*table.get_children())
    ax.clear()
    ax.imshow(map_img, interpolation="lanczos", origin="upper")

    response = stub.RequestProcessedData(messaging_pb2.ProcessedDataRequest(request="Request data"))

    for uuid_count, data in enumerate(response.processed_data):
        result = data
        split_str = result.split()
        lat = float(split_str[0])
        lon = float(split_str[1])
        rad = float(split_str[2])
        conf = float(split_str[3])

        x, y = map_.to_pixels(lat, lon)
        ax.scatter(x, y, s=rad, c="purple", alpha=0.5)

        table.insert(
            parent="",
            index="end",
            iid=uuid_count,
            text="",
            values=(str(uuid_count), str(lat), str(lon), str(rad), str(conf)),
        )
    
    table.yview_moveto(1)
    canvas.draw()
    root.after(6000, update_map)


update_map()
root.mainloop()
