# This file is about how to create a dataloader in pytorch way
from __future__ import print_function
import os
import numpy as np
import cv2
from PIL import Image
import argparse
from utils.utils import my_args_parser
import pdb

import torch
from torchvision import transforms
from torch.utils.data import Dataset

input_size = 112


def parse_oneline(line):
    line = line.strip().split()
    image_path = line[0]
    bbox = list(map(int, list(map(float, line[1:5]))))  # list: [x1, y1, x2, y2]
    x = list(map(float, line[5::2]))
    y = list(map(float, line[6::2]))
    landmarks = list(zip(x, y))  # list: [[x1,y1], [x2,y2], ...]
    return image_path, bbox, landmarks

def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels

def convert_landmarks(landmarks, input_size, orig_face_size):
    # Convert positions of landmarks(ndarrays) in sample to be consistent with image_after_Normalize.
    # Output: shape should be (42, ) not (21, 2)
    # Resize
    w, h = orig_face_size
    expand_w, expand_h = input_size / w, input_size / h
    landmarks *= np.array([expand_w, expand_h]).astype(np.float32)
    return landmarks

class Normalize(object):
    """
        Convert a PIL gray image to ndarrays
        Resize to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """

    def __call__(self, sample):
        img_crop, landmarks_orig = sample['image'], sample['landmarks']
        image = np.asarray(
            img_crop.resize((input_size, input_size), Image.BILINEAR),
            dtype=np.float32)  # Image.ANTIALIAS)
        # img normalize
        image = channel_norm(image)
        # gt normalize
        # landmarks = np.array(landmarks_orig).astype(np.float32)
        landmarks = convert_landmarks(landmarks_orig, input_size, img_crop.size)
        # Normalize
        landmarks = channel_norm(landmarks)
        # landmarks = landmarks.flatten()
        return {'image': image,
                'landmarks': landmarks
                }

class Rotation(object):
    """
        Input: resized_img (ndarrays); converted_landmarks (ndarrays with shape of (21,2))
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        angle = np.random.random_integers(-20, 20)
        mat = cv2.getRotationMatrix2D((image.shape[0]//2, image.shape[1]//2), angle, 1)
        image = cv2.warpAffine(image, mat, (image.shape[0], image.shape[1]))

        # landmarks = convert_landmarks(landmarks, input_size, orig_size)
        landmarks_rotate = []
        for landmark in landmarks:
            landmark = (mat[0][0]*landmark[0]+mat[0][1]*landmark[1]+mat[0][2],
                        mat[1][0]*landmark[0]+mat[1][1]*landmark[1]+mat[1][2])
            landmarks_rotate.append(landmark)
        return {'image': image,
                'landmarks': landmarks_rotate
                }

class Flip(object):
    """
        Input: ndarrays
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        return {'image': image,
                'landmarks': landmarks
                }

class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    # Actually the output here is only (C x H x W),
    # the batch_size "N" will be added automatically by torch.utils.data.DataLoader()

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        # change (21, 2) -> (42,)
        landmarks = np.array(landmarks).astype(np.float32)
        landmarks = landmarks.flatten()
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

class FaceLandmarksDataset(Dataset):
    def __init__(self, args, lines, transformer=None):
        self.args = args
        self.lines = lines
        self.transformer = transformer

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        '''
        :param index:
        :return single_sample: information dict of one single image
        '''

        image_path, bbox, landmarks = parse_oneline(self.lines[index])
        # print("img_path: ", image_path)
        img = Image.open(image_path).convert('L')  # gray_scale (for calculating mean & std)
        img_crop = img.crop(tuple(bbox))  # img.crop(tuple(x1,y1,x2,y2))

        if self.args.phase == 'Test' or self.args.phase == 'test':  # test only ToTensor()
            img_resize = np.asarray(
                img_crop.resize((input_size, input_size), Image.BILINEAR),
                dtype=np.float32)
            landmarks = convert_landmarks(landmarks, input_size, img_crop.size)
            single_sample = {
                'image': img_resize,
                'landmarks': landmarks
            }
            single_sample = self.transformer(single_sample)
        else:
            if self.args.no_normalize:  # train without normalize
                img_resize = np.asarray(
                    img_crop.resize((input_size, input_size), Image.BILINEAR),
                    dtype=np.float32)
                landmarks = convert_landmarks(landmarks, input_size, img_crop.size)
                single_sample = {
                    'image': img_resize,
                    'landmarks': landmarks
                }
                single_sample = self.transformer(single_sample)
            else:        # train with normalize
                single_sample = {
                    'image': img_crop,
                    'landmarks': landmarks
                }
                single_sample = self.transformer(single_sample)


        return single_sample

def load_data(args, phase):
    '''
    :param args:
    :return dataset: a instance of class FaceLandmarksDataset(), which will be used by torch.DataLoader
    '''

    if phase == 'Test' or phase == 'test':
        data_path = args.test_set_path
        with open(data_path, 'r') as f:
            lines = f.readlines()
        transformer = transforms.Compose([
            ToTensor()
        ])
    else:
        data_path = args.train_set_path
        with open(data_path, 'r') as f:
            lines = f.readlines()
        if not args.no_normalize:
            transformer = transforms.Compose([
                Normalize(),
                Rotation(),
                ToTensor()
            ])
        else:
            transformer = transforms.Compose([
                Rotation(),
                ToTensor()
            ])

    dataset = FaceLandmarksDataset(args, lines, transformer)
    return dataset


def dataloader(args):
    train_set = load_data(args, 'Train')
    valid_set = load_data(args, 'Test')
    return train_set, valid_set


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Face_Landmarks_Detection")
    args = my_args_parser(argparser)
    train_set = load_data(args, 'Test')
    for i in range(1, len(train_set)):
        sample = train_set[i]
        img = sample['image']
        landmarks = sample['landmarks']       # shape = (42,)

        # need to supress Normalize, since it will normalize the pixels to around 1 (not 0~255)
        img = np.squeeze(np.array(img, dtype=np.uint8))
        landmarks = landmarks.reshape(21, 2)
        for landmark in landmarks:
            landmark = np.uint8(landmark).tolist()
            img = cv2.circle(img, tuple(landmark), 3, (0, 0, 255), -1)
        cv2.imshow("face_landmarks", img)
        key = cv2.waitKey()
        if key == 27:
            exit(0)
        cv2.destroyAllWindows()
