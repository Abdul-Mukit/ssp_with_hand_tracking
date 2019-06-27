## Introduction
In this repository I have combined three functionalities to detect pose of target objects held by hand in dark.

Following are the repositories/ideas I have relied on:
1. [Single Shot Pose](https://github.com/Microsoft/singleshotpose) for tracking object pose. 
2. YOLO2 PyTorch impelementation by marvis ([pytorch-yolo2](https://github.com/marvis/pytorch-yolo2)) for tracking hands.
3. Gamma correction using OpenCV ([Link](https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/)) for processing
dark/low-exposure frames.

## Downloads
I uploaded the necessary weights in my [Dropbox](https://www.dropbox.com/sh/922jtluce1a1go3/AAAeDAbF-ZQ9JFV7aSMTY69Ga?dl=0).

Put the 'backup' folder in the project direcory. 

Mesh Folder is not shared publickly. Please email me for that as that is a private file. If you want to test with
any object other than 'cautery' please create a "Mesh" folder and put your .ply file inside it.

## Installation
Please follow instructions for [DOPE](https://github.com/NVlabs/Deep_Object_Pose).

The original pytorch-yolo2 was workable only with PyTorch version 0.3.1.
After my modifications it shuld run on 0.4.0 version as well.

## Usage:
I have used an Intel Realsense D435i camera for this impelematation as I needed to control exposure.

live_ssp_hand_realsense.py is the final impelmentation. In the code change the "Settings" section if needed.
Please watch my [demo video](https://youtu.be/XwVy5sZZxG8) for more details.

live_pose_detect_realsense.py and live_pose_detect_webcam.py are just SSP demos using the realsense camera and webcam respectively.




