# This is using Intel RealSense
import os
import time
import torch
from torch.autograd import Variable
import scipy.io
import warnings
import cv2

warnings.filterwarnings("ignore")
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

from darknet import Darknet
import dataset
import utils_ssp as ussp
from MeshPly import MeshPly




def valid(datacfg, cfgfile, weightfile):
    pose_conf_thresh = 0.1
    visualize = True

    test_width = 640
    test_height = 480
    linewidth = 1
    gpus = '0'  # Specify which gpus to use

    # Parse configuration files
    options = ussp.read_data_cfg(datacfg)
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
    model_pose = Darknet(cfgfile)
    model_pose.print_network()
    model_pose.load_weights(weightfile)
    model_pose.cuda()
    model_pose.eval()

    # RealSense Start
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # Setting exposure
    s = profile.get_device().query_sensors()[1]
    s.set_option(rs.option.exposure, 166)

    while (True):
        t1 = time.time()
        # Reading RGB image from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img = np.asanyarray(color_frame.get_data())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', img)
        # cv2.waitKey(1)

        Dataimg = np.transpose(img, (2, 0, 1))
        Dataimg = np.expand_dims(Dataimg, axis=0)
        data = torch.from_numpy(Dataimg).float()/255

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
        corners2D_pr[:, 0] = corners2D_pr[:, 0] * test_width
        corners2D_pr[:, 1] = corners2D_pr[:, 1] * test_height

        # Compute [R|t] by pnp
        R_pr, t_pr = ussp.pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),
                                  dtype='float32'), corners2D_pr,
                         np.array(internal_calibration, dtype='float32'))

        # Visualize
        Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
        proj_corners_pr = np.transpose(ussp.compute_projection(corners3D, Rt_pr, internal_calibration))


        t2 = time.time()
        ProcessingRate = 1 / (t2 - t1)
        # print('Current Frame Rate: {}'.format(FrameRate))

        if visualize:
            plt.xlim((0, test_width))
            plt.ylim((0, test_height))
            plt.imshow(scipy.misc.imresize(img, (test_height, test_width)))
            plt.ion()
            # Projections
            for edge in edges_corners:
                plt.plot(proj_corners_pr[edge, 0], proj_corners_pr[edge, 1], color='b', linewidth=linewidth)
            plt.gca().invert_yaxis()
            plt.pause(0.001)
            plt.clf()

        t3 = time.time()
        DisplayRate = 1 / (t3 - t1)
        print('SSP processing Rate: {}, Display rate: {}'.format(ProcessingRate, DisplayRate))



    cv2.destroyAllWindows()














datacfg = 'cfg/cautery.data'
cfgfile = 'cfg/yolo-pose.cfg'
weightfile = 'backup/cautery/model_backup1.weights'
valid(datacfg, cfgfile, weightfile)