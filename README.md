# RT-Flight

## Contributors

- [Allan Lago](https://github.com/alago1)
- [Sahaj Patel](https://github.com/sah4jpatel)
- [Matthew Clausen](https://github.com/matt-clausen)
- [Tom Liraz](https://github.com/tomliraz)
- [Tyler J. Schultz](https://github.com/tj-schultz)

## General File Structure

```
root/
├── data/  # Data for the simulator
│    ├── Blore_Clean.jpg
│    └── ...
├── yolo/  # Model weights and config
│    ├── yolov3-aerial.cfg
│    ├── yolov3-aerial.weights
│    └── ...
├── examples/  # Runnable examples
│    ├── util/
│    ├── mocked_realtime_uav.py
│    └── ...
├── server/  # Code usable in a Raspberry Pi
│    ├── engines/
│    ├── layers/
│    ├── models/
│    ├── tools/
│    │   ├── visualizer.py
│    │   ├── file_sender.py
│    │   └── ...
│    ├── main.py
│    ├── requirements.txt
│    └── ...
├── django_ui/  # Code for the UI
│    ├── django_ui/
│    ├── markers/
│    ├── manage.py
│    └── requirements.txt
└── ...
```

## Setup before using

You may want to add new images to the `data/` folder. The mocked simulation makes use of the Blore_Clean geotiff image which can be found [here](https://drive.google.com/file/d/1_EdDilrEKAX_WJgnPjyXAN751kO6PBq7/view?usp=sharing). You can also find the sequential image data (with exif metadata) [here](https://drive.google.com/file/d/1lOAdMvF40pLJ9tO97fJLUsY6I-lxEM43/view?usp=sharing). It can be used to run the 'sequential_dataset.py' example. Additionally, we make the use of [YoloV3 Darknet](https://github.com/jekhor/darknet) fine-tuned to aerial imagery as the detection model by default.

For this project we do not support the darknet model directly. You can find a keras h5 file converted from the darknet model [here](https://drive.google.com/file/d/1BlBvoZ2tIgFhMUnHYLhZB6-nmU6WrSHB/view?usp=sharing). We provide detailed instructions of how to convert from keras to tflite, edge-tpu, and onnx on `yolo/keras2tflite.ipynb` and `yolo/keras2onnx.ipynb`. Other formats, such as tensorrt, can be converted to from onnx.

<details>

<summary>
(Optional) Instructions if you want to convert the Darknet model yourself
</summary>

The weights for the Darknet model can be found here [here](https://drive.google.com/file/d/1LyWvsoPmmPM9is0TmDmCE5vddXZLXYK6/view?usp=share_link). Add them to the `yolo/` folder in the root directory. You will require "yolov3-aerial.weights" and "yolov3-aerial.cfg" (provided) files.

If you're using a different model (for instance [YoloV3-tiny](https://github.com/smarthomefans/darknet-test) on low-memory devices), please add the weights and config files
to the `yolo/` folder.

This [sample](https://github.com/NVIDIA/TensorRT/blob/main/samples/python/yolov3_onnx/yolov3_to_onnx.py) was helpful in converting the model to Onnx.

</details>

## **Installation**

Tested on: Ubuntu 22.04, Python 3.10

We recommend using a virtual environment to install the dependencies.

```bash
cd rt-flight/server
python3 -m venv venv
source venv/bin/activate
```

By default, we use GDAL to compute the gps coordinates but we also support a Geopy backend if you don't want to install that dependency on the server-side (client-side still needs it).

```bash
# install GDAL (optional on server-side)
apt install gdal-bin
apt install libgdal-dev
pip install gdal==3.7.0  # python bindings
```

Install server-side (`server/`) dependencies:

```bash
# install server-side dependencies
apt install libimage-exiftool-perl

# inside rt-flight/server
pip install -r 'requirements.txt'
```

In addition to the above dependencies, you must install a backend for running the detection model. The supported backends are `tflite`, `onnx`, `coral`, and `tensorrt`. Follow the instructions for the backend you want to use below.

<details>
<summary>
TensorFlow Lite
</summary>

```bash
pip install tflite_runtime
```

</details>
<details>
<summary>
Onnx GPU Runtime
</summary>

```bash
pip install onnruntime_gpu
```

</details>

<details>
<summary>
PyCoral
</summary>

```bash
pip install tflite_runtime

# https://coral.ai/docs/accelerator/get-started/#runtime-on-linux
apt install python3-pycoral
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

</details>

<details>
<summary>
TensorRT
</summary>

Varies from system to system. Please follow the instructions [here](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

</details>

### **Client-side (Ground Station)**

The client-side (`django_ui/`) is a GeoDjango application that requires additional dependencies. We recommend following the [GeoDjango installation guide](https://docs.djangoproject.com/en/3.2/ref/contrib/gis/install/#installing-geodjango) first. We've tested with the SpatialLite database.

```bash
# create client-side virtual env
cd rt-flight/django_ui
python3 -m venv venv
source venv/bin/activate

# install client-side dependencies
pip install -r 'requirements.txt'

# init database if not already done
python manage.py shell -c "import django;django.db.connection.cursor().execute('SELECT InitSpatialMetaData(1);')"
python manage.py migrate
```

---

## Running the code

Running the simulation requires running multiple programs. For single image files we use the `file_sender` script acts as an intermediary that emulates the transmission of a detection to the Ground Station and to the UI. We can also run a simulation of a flying uav with the `mocked_realtime_uav` script instead.

Run each of the following in a separate terminal.

### **Start the backend**

```bash
python -m server.main
```

### **Start the UI**

```bash
python django_ui/manage.py runserver
# you should now be able to access the UI on your browser at localhost:8000/markers/map
```

### **Run the intermediary script**

```bash
# for single image files (more akin to the real-world scenario)
python -m server.tools.file_sender -i <image file path>

# for simulated uav (images are generated from orthomosaic random path)
python examples/mocked_realtime_uav.py
```

## Run Production Code

`server` is a working version of the code that is running on CUSP, a custom-built thermal and RGB sensor package. More information about this project can be found [here](https://github.com/JesseChin/CUSP).
