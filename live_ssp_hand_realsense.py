import os
import time
import torch
from torch.autograd import Variable
import cv2

import utils_orgyolo as uyolo
import numpy as np
import pyrealsense2 as rs
import collections
from darknet import Darknet

import matplotlib.pyplot as plt
import scipy.misc
import utils_ssp as ussp
from MeshPly import MeshPly


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def do_pose_detect(img, use_cuda, img_width, img_height):
    Dataimg = np.transpose(img, (2, 0, 1))
    Dataimg = np.expand_dims(Dataimg, axis=0)
    data = torch.from_numpy(Dataimg).float() / 255

    # Pass data to GPU
    if use_cuda:
        data = data.cuda()

    data = Variable(data, volatile=True)

    # Forward pass
    output = model_pose(data).data

    # Using confidence threshold, eliminate low-confidence predictions
    all_boxes = ussp.get_region_boxes(output, pose_conf_thresh, num_classes)

    # For each image, get all the predictions
    boxes = all_boxes[1]

    # Iterate through each ground-truth object
    best_conf_est = -1

    # If the prediction has the highest confidence, choose it as our prediction for 1 object pose estimation
    for j in range(len(boxes)):
        if boxes[j][18] > best_conf_est:
            box_pr = boxes[j]
            best_conf_est = boxes[j][18]

    # Denormalize the corner predictions
    corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
    corners2D_pr[:, 0] = corners2D_pr[:, 0] * img_width
    corners2D_pr[:, 1] = corners2D_pr[:, 1] * img_height

    # Compute [R|t] by pnp
    R_pr, t_pr = ussp.pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),
                                   dtype='float32'), corners2D_pr,
                          np.array(internal_calibration, dtype='float32'))

    # Visualize
    Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
    proj_corners_pr = np.transpose(ussp.compute_projection(corners3D, Rt_pr, internal_calibration))
    return proj_corners_pr

def crop_image(image, center, newSize, plot=False):
    x_center, y_center = center
    w_new, h_new = newSize
    h_org, w_org, ch = image.shape

    x_start = x_center - w_new/2
    x_end = x_center + w_new/2

    y_start = y_center - h_new/2
    y_end = y_center + h_new/2

    if x_start < 0:
        x_end   += abs(x_start)
        x_start += abs(x_start)
    elif x_end > w_org:
        x_start -= x_end-w_org
        x_end   -= x_end-w_org-1

    if y_start < 0:
        y_end   += abs(y_start)
        y_start += abs(y_start)
    elif y_end > h_org:
        y_start -= y_end-h_org
        y_end   -= y_end-h_org-1

    img_cropped = image[y_start:y_end, x_start:x_end, :]

    if plot:
        img_plot = image.copy()
        cv2.rectangle(img_plot, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        return img_cropped, [x_start, x_end, y_start, y_end], img_plot
    else:
        return img_cropped, [x_start, x_end, y_start, y_end]



#######################################################
# Settings
#######################################################
exposure_val = 25 # 166 is the default exposure value.
use_hand_tracking = True
gamma_correction = True

hand_crop_size = [200, 200]


pose_conf_thresh = 0.1
hand_conf_thresh = 0.6
gamma_val = 2


test_width = 640
test_height = 480
fps = 30
linewidth = 1
gpus = '0'  # Specify which gpus to use
use_cuda = True

datacfg = {'hands':   'cfg/hands.data',
           'cautery': 'cfg/cautery.data'}

cfgfile = {'hands':   'cfg/yolo-hands.cfg',
           'cautery': 'cfg/yolo-pose.cfg'}

weightfile = {'hands':   'backup/hands/000500.weights',
              'cautery': 'backup/cautery/model_backup1.weights'}

namesfile = {'hands': 'data/hands.names'}

#######################################################
# Setting up YOLO-hand
#######################################################
# Setting up YOLO
model_hand = Darknet(cfgfile['hands'])
model_hand.load_weights(weightfile['hands'])
print('Loading weights from %s... Done!' % (weightfile['hands']))

class_names = uyolo.load_class_names(namesfile['hands'])
if use_cuda:
    model_hand.cuda()

#######################################################
# Setting up SSP
#######################################################
options = ussp.read_data_cfg(datacfg['cautery'])
meshname = options['mesh']

# Parameters
seed = int(time.time())

torch.manual_seed(seed)
use_cuda = True
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)
num_classes = 1
edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

# Read object model information, get 3D bounding box corners
mesh = MeshPly(meshname)
vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
corners3D = ussp.get_3D_corners(vertices)

# Read intrinsic camera parameters
internal_calibration = ussp.get_camera_intrinsic()

# Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
model_pose = Darknet(cfgfile['cautery'])
model_pose.load_weights(weightfile['cautery'])
model_pose.cuda()
model_pose.eval()

#######################################################
# Setting up Intel RealSense Camera
#######################################################
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, test_width, test_height, rs.format.bgr8, fps)
profile = pipeline.start(config)
# Setting exposure
s = profile.get_device().query_sensors()[1]
s.set_option(rs.option.exposure, exposure_val)

#######################################################
# Running camera and processing
#######################################################
while True:
    # Reading image from camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue
    img = np.asanyarray(color_frame.get_data())
    # draw_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if gamma_correction:
        img = adjust_gamma(img, gamma=gamma_val)

    # YOLO stuff
    if use_hand_tracking:
        sized = cv2.resize(img, (model_hand.width, model_hand.height))
        bboxes = uyolo.do_detect(model_hand, sized, hand_conf_thresh, 0.4, use_cuda)
        if any(bboxes):
            center = [int(bboxes[0][0] * test_width), int(bboxes[0][1] * test_height)]
            img_hand_cropped, crop_box = crop_image(img, center, hand_crop_size)
            img_detection = img_hand_cropped
            # print(crop_box[3])
        else:
            img_detection = img
    else:
        img_detection = img

    ## Detecing object pose
    proj_corners_pr = do_pose_detect(img_detection, use_cuda, img_detection.shape[1], img_detection.shape[0])


    # Display
    draw_img = img_detection
    # draw_img = cv2.cvtColor(img_detection, cv2.COLOR_BGR2RGB)
    plt.xlim((0, draw_img.shape[1]))
    plt.ylim((0, draw_img.shape[0]))
    plt.imshow(scipy.misc.imresize(draw_img, (draw_img.shape[0], draw_img.shape[1])))
    plt.ion()
    for edge in edges_corners:
        plt.plot(proj_corners_pr[edge, 0], proj_corners_pr[edge, 1], color='b', linewidth=linewidth)
    plt.gca().invert_yaxis()
    plt.pause(0.001)
    plt.clf()






