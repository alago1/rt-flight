# RT-Flight

## Contributors

- [Allan Lago](https://github.com/alago1)
- [Sahaj Patel](https://github.com/sah4jpatel)
- [Matthew Clausen](https://github.com/matt-clausen)
- [Tom Liraz](https://github.com/tomliraz)
- [Tyler J. Schultz](https://github.com/tj-schultz)

## Before using

You may want to add a `/data` folder at the root with the appropriate images. The model_sender notebook makes use of the Blore_Clean geotiff image which can be found [here](https://drive.google.com/file/d/14mJcI-_crVwy95-pAr1K8nYJDyK99zh5/view?usp=share_link). Additionally, we make the use of YoloV3 Darknet as the detection model for our simulation. The weights for these models can be found here [here](https://github.com/jekhor/darknet) in the readme. You may also want to add a `/weights` folder in the `/raspberry_pi_code` directory and add in the "yolov3-aerial.weights" and "yolov3-aerial.cfg" files.

### **Install Dependencies**

OS: Ubuntu 22.04.2
Python version: 3.10.6

```
pip install -r 'requirements.txt'

apt install libimage-exiftool-perl
apt install python3-tk
apt install libgl1
apt install libglib2.0-0
```

***


## Using the Simulator

Running the simulation requires a specific order in which to launch the programs in. The network.py script acts as an intermediary that emulates the transmission of a detection via Mavlink to the Ground Station and to the UI. As such, network.py would be started first, followed by data_reciever.py to start up the UI. Finally, the model_sender.ipynb notebook will be run. 


### **Start the Network**

```
python realtime_ui/network.py
```

### **Start the UI**

Create a new terminal/command line instance to launch the UI. 

```
python realtime_ui/data_receiver.py
```

### **Run the Model_Sender Notebook**

While we don't have a great way to run a Jupyter notebook from command line, the "Run All" button within **model_sender.ipynb** would be adequate to start the simulator.


### **Example Run for Simulator**

```
#in one terminal instance
python realtime_ui/network.py
#in another terminal instance
python realtime_ui/data_receiver.py

# once both are running, run the full model_sender.ipynb notebook
```

***

## Run Production Code

This is the code within the `/raspberry_pi_code` directory that serves as a working version of the code that is running on CUSP, a custom-built thermal and RGB sensor package. More information about this project can be found [here](https://github.com/JesseChin/CUSP).

### **Run the Model Network

Similarly to the simulator, model_network.py works like network.py, with the difference being that it awaits a file path input from the use in the form of a procedure call, then runs the detection model on it and sends back the corresponding coordinates and information. By default, this server will be running on port 50051, and the log file location will be in the same directory as model_network.py named "log.txt".

```
python raspberry_pi_code/model_network <port number; default = 50051> <log file path; default = "log.txt">
```

### **Run the File Sender**

Once the model network is running, you can run detections on it by using the file_sender.py script to send an image file path and print out the resulting coordinates and information. Make sure to change the "img_path" variable within the script to your desired image to ensure that the program works as intended. 

```
python raspberry_pi_code/file_sender.py
```

### **Full Example**

```
#Run model_network.py on port 1004 and set logs to the root directory
python raspberry_pi_code/model_network.py 1004 "../logs.txt"

#Run file sender in a separate terminal instance
python raspberry_pi_code/file_sender.py
```

***

## Compile gRPC Protobuf (if needed)

In the case of issues with gRPC and the protos files, you can use the following to recompile the proto files for use within the model_network.py program:

`cd raspberry_pi_code/protos && python -m grpc_tools.protoc -I../protos --python_out=. --pyi_out=. --grpc_python_out=. messaging.proto`

***

## Warning

Graceful termination has not been properly integrated into network.py nor model_network.py so ensure that the process corresponding to either of these is properly killed, as opposed to suspended. Failure to do so, will cause issues with establishing a gRPC connection between the other components to the respective network program.


***

## Known Bugs

Currently, the radius value that is returned from the GPSTranslocationLayer in the raspberry_pi_code/model_network.py program is inaccurate, as it overshoots the known-value consistently.