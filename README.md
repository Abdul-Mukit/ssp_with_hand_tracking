## Introduction
In this repository I have combined three functionalities to detect pose of target objects held by hand in dark.
Please watch my [demo video](https://youtu.be/XwVy5sZZxG8) for more details.

Following are the repositories/ideas I have relied on:
1. [Single Shot Pose](https://github.com/Microsoft/singleshotpose) for tracking object pose. 
2. YOLO2 PyTorch implementation by marvis ([pytorch-yolo2](https://github.com/marvis/pytorch-yolo2)) for tracking hands.
3. Gamma correction using OpenCV ([Link](https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/)) for processing
dark/low-exposure frames.

### Problem Description
My objective in this project is to detect a medical tool called a 
cautery in surgical lighting. A cautery is a electronic knife like tool
that looks like a pen. It is used to make cuts during surgery and is a
very extensively used tool.

<img
src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSlcHBUCwIrRFSLxGSz7Ss9m2lp9JIoPZ-uDrmKq_o9VnWj4bHqDg"
alt="Cautery tool" height="200"><img
src="https://www.anesplus.com/wp-content/uploads/2018/02/live-surgical-operation.jpg"
alt="Surgical Lighting" height="200">

As part of my research on Augmented Surgical Reality, I investigate the
usability of existing 6DoF object pose estimation methods such as Deep
Object Pose Estimation
[(DOPE)](https://github.com/NVlabs/Deep_Object_Pose) and Single Shot
Pose [(SSP)](https://github.com/microsoft/singleshotpose) for cautery
tool detection. After applying DOPE and SSP, I realized that the cautery
is a very difficult object to track for multiple reasons. Most important
reasons are: the cautery is very small and thin, featureless, is bound
to be heavily occluded. One other problem that is unique to Operation
Room environment only, is the high-intensity surgical lighting above. To
handle this, I tried lowering the exposure of the camera. However, due
to such high intensity of the lighting the exposure has to be lowered to
such a low level that neither YOLO2 nor DOPE is able to detect anything
[(DOPE in extreme lighting demo video)](https://youtu.be/rf-Hnc4QBsk).
This all makes it very challenging to track a cautery pen in operation
room like environment.

In order to overcome these problems in this project, I tried out merging
3 different methods. To rectify the dark image (due to low exposure
settings of camera) I added a gamma correction methods which makes the
image brighter. Following are two captured images before and after the
gamma correction respectively. Notice how the image gets brigher after
gamma correction. The performance jump due to gamma correction is very
clear. [A demonstration of the gamma correction and how
it helps tracking hands in the dark is demonstrated in this video](https://youtu.be/Khy8U_zXDC4).

<img 
src="https://github.com/Abdul-Mukit/dope_with_hand_tracking/raw/master/before_gamma.jpg"
height="300"><img
src="https://github.com/Abdul-Mukit/dope_with_hand_tracking/raw/master/after_gamma.jpg"
height="300">

I trained YOLO2 to track hands. Following is a demo image of hand
tracking using YOLO2. I trained my own weights for hand tracking instead
of using the existing hand-trackers. That is because **existing
hand-trackers track only bare hands without gloves**. If there are
gloves, then the trackers fail. My tracker can work with different
colored hands and will different gloves.

![yolo hand demo](https://github.com/Abdul-Mukit/dope_with_hand_tracking/raw/master/yolo_hand_demo.jpg)

Finally, I take a cropped square image around the hand which is then run
through SSP to identify the pose of the pen. By doing I eliminated
several false detections. SSP with hand tracking looks something like
the following image in normal lighting. The green box is centered at the
detected hand and the portion inside the green box is being cropped. The
yellow 3D bounding box shows the detected pose of the target
object.<br/>
![Final Result](https://github.com/Abdul-Mukit/dope_with_hand_tracking/raw/master/final_result_image.png)
<br/>

### Conclusion
After my experiments, I realized that I lacked proper dataset to train
SSP for being able to detect objects in the dark. I didn't get any
tracking even with the help of hand tracking. I demonstrate this using
DOPE through the following image where I try to detect the well known
[ycb-object](http://www.ycbbenchmarks.com/object-models/) 'spam can'. I
chose DOPE as it's more accurate than SSP and if DOPE can't detect
objects in dark so can't SSP. I used the original weights provided by
the authors of DOPE but still DOPE was not able to detect the can even
with the help of hand detection. This is because the lighting condition
used by the authors are not as extreme as mine( Please take a look at
the
[Falling Objects dataset](https://research.nvidia.com/publication/2018-06_Falling-Things)).
Currently I am working on making better dataset.

![Final demo using Spam can](https://github.com/Abdul-Mukit/dope_with_hand_tracking/raw/master/final_result_image_spam.png)

## Downloads
I uploaded the necessary weights in my [Dropbox](https://www.dropbox.com/sh/922jtluce1a1go3/AAAeDAbF-ZQ9JFV7aSMTY69Ga?dl=0).

Put the 'backup' folder in the project directory. 

Mesh Folder is not shared publicly. Please email me for that as that is
a private file. If you want to test with any object other than 'cautery'
please create a "Mesh" folder and put your .ply file inside it.

## Installation
Please follow instructions for [DOPE](https://github.com/NVlabs/Deep_Object_Pose).

The original pytorch-yolo2 was workable only with PyTorch version 0.3.1.
After my modifications it should run on 0.4.0 version as well.

## Usage:
I have used an Intel Realsense D435i camera for this implementation as I
needed to control exposure.

live_ssp_hand_realsense.py is the final implementation. In the code
change the "Settings" section if needed. Please watch my
[demo video](https://youtu.be/XwVy5sZZxG8) for more details.

live_pose_detect_realsense.py and live_pose_detect_webcam.py are just
SSP demos using the realsense camera and webcam respectively.




