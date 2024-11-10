# SmartClassroom: A system with speaker tracking and live stream pushing capabilities

This SmartClassroom System is designed for online and offline integrated teaching courses. The classroom system uses multiple cameras for target tracking to reduce the delay of information transmission, improve the efficiency of classroom interaction, and provide video information with lower delay and higher recognition rate. In this way, we can better record each on-site student's speech status, almost without manual operation. Also, online students can feel a more real learning experience. At the same time, the system can put courses on the live streaming platform through live streaming, allowing more people to participate in online teaching. 



## Speaker Tracking

The function is divided into two modules: rough positioning based on microphone beamforming principle and fine positioning based on YOLOv5 algorithm.

These two parts can operate independently: rough positioning requires devices such as array microphones and webcams, the former will output the approximate area of the sound source, while the camera will turn to the approximate position without accuracy; Fine positioning only requires the camera, the face coordinates will be obtained through the YOLOv5 algorithm, and the camera rotation will be controlled based on PID so that the face is located in the center of the picture, but this requires the face itself to be in the picture. Two terminals can also be opened to run at the same time, which can make the face that is not in the picture first through rough positioning to be found, and then through fine positioning to be accurately aligned.

You need to replace the IP in the code with the IP of your own device.

### Rough Positioning

#### Installation

```bash
python -m venv rough_positioning
source rough_positioning/bin/activate
pip install -r requirements_rough.txt
```

#### Start positioning

```bash
bash activate_rough.sh
```

### Fine Positioning

#### Installation

```bash
conda create --name fine_positioning python=3.10
conda activate fine_positioning
pip install -r requirements_fine.txt
```

#### Start positioning

```bash
bash activate_fine.sh
```



## Live Stream Pushing

Coming soon ...



## Acknowledgments 

Thanks to [Ultralytics](https://github.com/ultralytics) for providing [yolov5](https://github.com/ultralytics/yolov5), which has been a significant support for our project.