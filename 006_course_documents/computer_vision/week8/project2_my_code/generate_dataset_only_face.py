# This program is for selecting valid data from the original dataset, and generating train.txt & test.txt

import numpy as np
import os
import cv2
import argparse
from utils.utils import my_args_parser
import pdb

class Generate_dataset(object):
    def __init__(self, args):
        self.folder_list = ['I', 'II']
        self.expand_ratio = 0.25
        self.out_path = 'data'
        self.split_ratio = 0.1   # train:test = 9:1
        self.args = args

    def remove_invalid_image(self, lines):
        images = []
        for line in lines:
            name = line.split()[0]
            if os.path.isfile(name):
                images.append(line)
        return images

    def load_metadata(self):
        tmp_lines = []
        for folder_name in self.folder_list:
            folder = os.path.join('data', folder_name)
            metadata_file = os.path.join(folder, 'label.txt')
            with open(metadata_file) as f:
                lines = f.readlines()
            tmp_lines.extend(list(map((folder + '/').__add__, lines)))
        res_lines = self.remove_invalid_image(tmp_lines)
        return res_lines

    def show_gt_image(self):
        # check if the transformed gt_images are right or not
        res_lines = self.load_metadata()
        for line in res_lines:
            line = line.strip().split()
            image_path = line[0]
            x1, y1, x2, y2 = list(map(int, list(map(float, line[1:5]))))
            bbox = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            x_landmark = list(map(int, list(map(float, line[5::2]))))
            y_landmark = list(map(int, list(map(float, line[6::2]))))
            landmarks = list(zip(x_landmark, y_landmark))  # a python list

            # plot bbox and landmarks on image
            img = cv2.imread(image_path)
            img = cv2.polylines(img, [np.int32(bbox)], True, (0, 255, 255), 3, cv2.LINE_AA)
            for landmark in landmarks:
                img = cv2.circle(img, landmark, 5, (0,0,255), 0)
            cv2.imshow("face_landmarks", img)
            key = cv2.waitKey()
            if key == 27:
                exit(0)
            cv2.destroyAllWindows()

    def get_single_line_truth(self, line):
        line = line.strip().split()
        name = line[0]
        rect = list(map(int, list(map(float, line[1:5]))))    # list: [x1, y1, x2, y2]
        x = list(map(float, line[5::2]))
        y = list(map(float, line[6::2]))
        landmarks = list(zip(x, y))
        return name, rect, landmarks

    def expand_roi(self, img_name, rect):
        ratio = self.expand_ratio
        img = cv2.imread(img_name, 0)
        img_h, img_w = img.shape

        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        face_w = x2 - x1 + 1
        face_h = y2 - y1 + 1
        expanded_w = int(face_w * ratio)
        expanded_h = int(face_h * ratio)
        new_x1 = x1 - expanded_h
        new_x2 = x2 + expanded_h
        new_y1 = y1 - expanded_w
        new_y2 = y2 + expanded_w
        new_x1 = 0 if new_x1 < 0 else new_x1
        new_x2 = int(img_w-1) if new_x2 >= img_w else new_x2
        new_y1 = 0 if new_y1 < 0 else new_y1
        new_y2 = int(img_h-1) if new_y2 >= img_h else new_y2

        expanded_roi = [new_x1, new_y1, new_x2, new_y2]
        return expanded_roi

    def generate_train_test_set(self):
        # Some landmarks will be minus finally, so throw it when using it.
        # Because if use it as 0, the landmark will be a different position in the face.(we do not need it)

        res_lines = self.load_metadata()
        truth = []   # Use {} when generating .json format
        for line in res_lines:
            name, rect, landmarks = self.get_single_line_truth(line)
            expanded_roi = self.expand_roi(name, rect)
            landmarks -= np.array([expanded_roi[0], expanded_roi[1]])
            # For .json generation
            # if name not in truth:
            #     truth[name] = []
            # truth[name].append((expanded_roi, landmarks))
            landmarks = landmarks.flatten().tolist()
            truth.append([name, expanded_roi, landmarks])

        indices = np.arange(len(truth))
        np.random.shuffle(indices)
        split_point = int(self.split_ratio * len(truth))
        test_indices, train_indices = np.split(indices, (split_point,))

        train_outfile = os.path.join(self.out_path, 'train.txt')
        with open(train_outfile, 'w') as outfile:
            for i in train_indices:
                outfile.write(truth[i][0] + " ")
                for bbox_pos in (list(map(str, truth[i][1]))):
                    outfile.write(bbox_pos + " ")
                for landmarks_pos in (list(map(str, truth[i][2]))):
                    outfile.write(landmarks_pos + " ")
                outfile.write('\n')

        test_outfile = os.path.join(self.out_path, 'test.txt')
        with open(test_outfile, 'w') as outfile:
            for i in test_indices:
                outfile.write(truth[i][0] + " ")
                for bbox_pos in (list(map(str, truth[i][1]))):
                    outfile.write(bbox_pos + " ")
                for landmarks_pos in (list(map(str, truth[i][2]))):
                    outfile.write(landmarks_pos + " ")
                outfile.write('\n')

    def inspect_output(self, inspect_txt_path=None):
        # For used in detector_myself.py
        if inspect_txt_path:
            metadata_file = inspect_txt_path
        else:
            metadata_file = self.args.test_set_path

        with open(metadata_file) as f:
            res_lines = f.readlines()

        for line in res_lines:
            line = line.strip().split()
            image_path = line[0]
            x1, y1, x2, y2 = list(map(int, list(map(float, line[1:5]))))
            x_landmark = list(map(int, list(map(float, line[5::2]))))
            y_landmark = list(map(int, list(map(float, line[6::2]))))
            landmarks = list(zip(x_landmark, y_landmark))  # a python list

            # plot bbox and landmarks on image
            img = cv2.imread(image_path)
            img_crop = img[y1:y2, x1:x2, :]
            for landmark in landmarks:
                img_crop = cv2.circle(img_crop, landmark, 5, (0, 0, 255), 0)
            cv2.imshow("face_landmarks", img_crop)
            key = cv2.waitKey()
            if key == 27:
                exit(0)
            cv2.destroyAllWindows()


def main():
    my_parser = argparse.ArgumentParser(description="Face_Landmarks_Detection")
    args = my_args_parser(my_parser)
    gen_data = Generate_dataset(args)
    # gen_data.generate_train_test_set()
    # gen_data.show_gt_image()   # check the original gt on image
    gen_data.inspect_output()    # check the final train/test data on image

if __name__ == '__main__':
    main()