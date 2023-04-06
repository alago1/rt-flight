import smopy
import tkinter as tk
from tkinter import *
from tkinter import ttk
import grpc
import os, sys
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as mcolors
import matplotlib
import mplcursors
from mplcursors import _pick_info

sys.path.append(os.path.join(os.path.dirname(__file__), "protos"))
import messaging_pb2
import messaging_pb2_grpc

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from sklearn.cluster import DBSCAN

channel = grpc.insecure_channel('localhost:50051')
stub = messaging_pb2_grpc.MessagingServiceStub(channel)

root = tk.Tk()
root.geometry("800x1100")

map_updated = False

fig = Figure(frameon=False)
ax = fig.add_subplot(111)
ax.axis("off")

def format_tooltip_text(uuid, cluster_label, lat, lon, rad, conf, cluster_size):
    lat = "{:.7f}".format(lat)
    lon = "{:.7f}".format(lon)
    rad = "{:.2f}".format(rad)
    return (
        f"UUID: {uuid}\n"
        f"Cluster: {cluster_label}\n"
        f"Coords: {lat}, {lon}\n"
        f"Radius: {rad}m\n"
        f"Confidence: {int(conf)}\n"
        f"Cluster size: {cluster_size}"
    )

def show_waiting_text():
    waiting_img = Image.new('RGBA', (1024, 1024), (255, 255, 255, 255))
    draw = ImageDraw.Draw(waiting_img)
    text = "Waiting for coordinates..."
    font = ImageFont.truetype("Helvetica", 24)
    text_width, text_height = draw.textsize(text, font)
    x = (waiting_img.width - text_width) // 2
    y = (waiting_img.height - text_height) // 2
    draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))
    return waiting_img

waiting_img = show_waiting_text()
ax.imshow(waiting_img, interpolation="lanczos", origin="upper")
ax.axis("off")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

table_view = tk.Frame(root)
table_view.pack()

table = ttk.Treeview(table_view)
yscrollbar = ttk.Scrollbar(table_view, orient="vertical", command=table.yview)
yscrollbar.pack(side=tk.RIGHT, fill=tk.Y)
table.configure(yscrollcommand=yscrollbar.set)

table["columns"] = ("UUID", "lat", "long", "radius", "conf", "cluster")

table.column("#0", width=0, stretch=tk.YES)
table.column("UUID", anchor=tk.CENTER, width=75)
table.column("lat", anchor=tk.CENTER, width=100)
table.column("long", anchor=tk.CENTER, width=100)
table.column("radius", anchor=tk.CENTER, width=100)
table.column("conf", anchor=tk.CENTER, width=75)
table.column("cluster", anchor=tk.CENTER, width=100)

table.heading("#0", text="", anchor=tk.CENTER)
table.heading("UUID", text="UUID", anchor=tk.CENTER)
table.heading("lat", text="lat", anchor=tk.CENTER)
table.heading("long", text="long", anchor=tk.CENTER)
table.heading("radius", text="radius (m)", anchor=tk.CENTER)
table.heading("conf", text="conf", anchor=tk.CENTER)
table.heading("cluster", text="cluster", anchor=tk.CENTER)

table.pack()

uuid_count = 0
old_len = -1

def process_response_data(data, eps=0.0005, min_samples=3):
    if not data:
        return [], []
    samples = np.array([[float(x) for x in row.split()] for row in data])
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(samples[:, :2])
    return samples, labels

def update_map():
    global uuid_count
    global map_updated
    global map_
    global old_len
    global new_cmap
    global assigned_colors

    current_scroll_position = table.yview()[0]
    table.delete(*table.get_children())


    if not map_updated:
        ax.clear()
        ax.imshow(waiting_img, interpolation="lanczos", origin="upper")
        ax.axis("off")

    response = stub.RequestProcessedData(messaging_pb2.ProcessedDataRequest(request="Request data"))
    data, cluster_labels = process_response_data(response.processed_data)


    if not map_updated and len(data) > 0:
        map_updated = _extracted_from_update_map_15(data)

    pixel_locations = []

    if map_updated:
        
        for data_point in data:
            lat, lon, rad, conf = data_point
            x, y = map_.to_pixels(lat, lon)
            pixel_locations.append((x, y))

        if len(data) != old_len:
            old_len=len(data)
            pixel_locations = np.array(pixel_locations)
            if len(pixel_locations) > 0:
                ax.scatter(pixel_locations.T[0], pixel_locations.T[1], s=data[:, 2], alpha=0.05, linewidths=15, edgecolors="black")
                scatter_plot = ax.scatter(pixel_locations.T[0], pixel_locations.T[1], s=data[:, 2], c=cluster_labels, cmap="Spectral", alpha=0.5, linewidths=0.25, edgecolors="black")

                assigned_colors = scatter_plot.to_rgba(cluster_labels)

                annot = ax.annotate("", xy=(0, 0), xytext=(20, 20),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
                annot.set_visible(False)

                def update_annot(ind):
                    pos = scatter_plot.get_offsets()[ind["ind"][0]]
                    annot.xy = pos
                    uuid = ind["ind"][0]
                    cluster_label = cluster_labels[uuid]
                    lat, lon, rad, conf = data[uuid]
                    cluster_size = np.count_nonzero(cluster_labels == cluster_label)
                    text = format_tooltip_text(uuid, cluster_label, lat, lon, rad, conf, cluster_size)
                    annot.set_text(text)

                def hover(evt):
                    vis = annot.get_visible()
                    if evt.inaxes == ax:
                        cont, ind = scatter_plot.contains(evt)
                        if cont:
                            update_annot(ind)
                            annot.set_visible(True)
                            fig.canvas.draw_idle()
                        else:
                            if vis:
                                annot.set_visible(False)
                                fig.canvas.draw_idle()

                fig.canvas.mpl_connect("motion_notify_event", hover)

        for uuid_count, data_point, assigned_color in zip(range(len(data)), data, assigned_colors):
            lat, lon, rad, conf = data_point
            color_hex = mcolors.to_hex(assigned_color)
            table.tag_configure(color_hex, background=color_hex)

            table.insert(
                parent="",
                index="end",
                iid=uuid_count,
                text="",
                values=[str(v) for v in (str(uuid_count), "{:.7f}".format(lat), "{:.7f}".format(lon), "{:.2f}".format(rad), int(conf), cluster_labels[uuid_count] if cluster_labels[uuid_count] != -1 else "Noise")],
                tags=(color_hex,),
            )

        is_scrollbar_at_bottom = current_scroll_position >= 1
        if is_scrollbar_at_bottom:
            table.yview_moveto(1)
        else:
            # Maintain the current scrollbar position
            table.yview_moveto(current_scroll_position)

    canvas.draw()
    root.after(1000, update_map)

def _extracted_from_update_map_15(data):
    global result, map_img, lat_min, lat_max, lon_min, lon_max, map_
    lat_center = data[:, 0].mean()
    lon_center = data[:, 1].mean()

    lat_mile = 0.0144927536231884
    lon_mile = 0.0181818181818182
    lat_min = lat_center - (0.11 * lat_mile)
    lat_max = lat_center + (0.11 * lat_mile)
    lon_min = lon_center - (0.11 * lon_mile)
    lon_max = lon_center + (0.11 * lon_mile)
        
    map_ = smopy.Map((lat_min, lon_min, lat_max, lon_max), z=17)

    map_img = map_.to_pil()

    ax.clear()
    ax.imshow(map_img, interpolation="lanczos", origin="upper")
    result = True
    return result

update_map()
root.mainloop()

