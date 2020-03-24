# This program will use the trained model to predict and show/save the predicted images
import os
import cv2
from PIL import Image
import numpy as np
import torch

from utils import utils

def predictor(args, model, device):
    # case1: img_path is a directory
    if os.path.isdir(args.img_path):
        img_names = []
        lists = os.listdir(args.img_path)
        for file_name in sorted(lists):
            img_names.append(os.path.join(args.img_path, file_name))
    # case2: img_path is a single file
    else:
        img_names = [args.img_path]

    # load model
    utils.load_state(args.trained_model_path, model)
    model.to(device)

    with torch.no_grad():
        for img_name in img_names:
            # process img
            img_orig = cv2.imread(img_name)
            img_resize_1 = cv2.resize(img_orig,
                                    (int(args.input_size), int(args.input_size)), interpolation=cv2.INTER_NEAREST)
            img_resize_2 = img_resize_1.astype(np.float32)
            img_gray = cv2.cvtColor(img_resize_2, cv2.COLOR_BGR2GRAY)
            input_img = np.expand_dims(img_gray, axis=0)
            input_img = np.expand_dims(input_img, axis=0)
            input_img = torch.from_numpy(input_img).to(device)
            output = model(input_img)
            output = output.detach().cpu().numpy()[0].tolist()

            x = list(map(int, output[0: len(output): 2]))
            y = list(map(int, output[1: len(output): 2]))
            landmarks_output = list(zip(x, y))

            for landmarks in landmarks_output:
                cv2.circle(img_resize_1, tuple(landmarks), 2, (0, 255, 255), -1)
            # show img in full screen
            cv2.namedWindow(str(img_name), cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(str(img_name), cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(str(img_name), img_resize_1)
            key = cv2.waitKey()
            if key == 27:
                exit()
            cv2.destroyAllWindows()
