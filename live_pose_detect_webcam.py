# %matplotlib inline
import os
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
import warnings
import cv2

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import scipy.misc

from darknet import Darknet
import dataset
from utils_ssp import *
from MeshPly import MeshPly




def valid(datacfg, cfgfile, weightfile):
    conf_thresh = 0.1
    visualize = True

    test_width = 640
    test_height = 480
    linewidth = 1
    gpus = '0'  # Specify which gpus to use

    # Parse configuration files
    options = read_data_cfg(datacfg)
    valid_images = options['valid']
    meshname = options['mesh']
    backupdir = options['backup']
    name = options['name']

    # Parameters
    prefix = 'results'
    seed = int(time.time())

    torch.manual_seed(seed)
    use_cuda = True
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        torch.cuda.manual_seed(seed)



    use_cuda = True
    num_classes = 1
    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]

    # Read object model information, get 3D bounding box corners
    mesh = MeshPly(meshname)
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D = get_3D_corners(vertices)

    # Read intrinsic camera parameters
    internal_calibration = get_camera_intrinsic()

    # Specicy model, load pretrained weights, pass to GPU and set the module in evaluation mode
    model = Darknet(cfgfile)
    model.print_network()
    model.load_weights(weightfile)
    model.cuda()
    model.eval()

    # # Capture frame-by-frame
    cap = cv2.VideoCapture(0)

    while (True):
        t1 = time.time()
        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Dataimg = np.transpose(img, (2, 0, 1))
        Dataimg = np.expand_dims(Dataimg, axis=0)
        data = torch.from_numpy(Dataimg).float()/255

        # Pass data to GPU
        if use_cuda:
            data = data.cuda()

        # Wrap tensors in Variable class, set volatile=True for inference mode and to use minimal memory during inference
        data = Variable(data, volatile=True)

        # Forward pass
        output = model(data).data

        # Using confidence threshold, eliminate low-confidence predictions
        all_boxes = get_region_boxes(output, conf_thresh, num_classes)


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
        R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), corners3D[:3, :]), axis=1)),
                                  dtype='float32'), corners2D_pr,
                         np.array(internal_calibration, dtype='float32'))

        # Visualize
        Rt_pr = np.concatenate((R_pr, t_pr), axis=1)
        proj_corners_pr = np.transpose(compute_projection(corners3D, Rt_pr, internal_calibration))


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



    cap.release()
    cv2.destroyAllWindows()














datacfg = 'cfg/cautery.data'
cfgfile = 'cfg/yolo-pose.cfg'
weightfile = 'backup/cautery/model_backup1.weights'
valid(datacfg, cfgfile, weightfile)