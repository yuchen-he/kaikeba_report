import cv2
import matplotlib.pyplot as plt
import numpy as np


class Object_tracking(object):
    def __init__(self, tolerate):
        self.sift = cv2.xfeatures2d.SURF_create()

        # parameters for draw bbox
        self.box_color = (0, 255, 255)
        self.box_thickness = 3

        # parameters for ransac
        self.tolerate = tolerate

    def draw_bbox(self, img, bbox):
        # draw a bbox on to a image (in BGR format)
        # return: image with bbox
        #img_bbox = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 5)
        #print('first_frame: ', img.shape)
        img_bbox = cv2.polylines(img, [np.int32(bbox)], True, self.box_color, self.box_thickness, cv2.LINE_AA)
        return img_bbox


    def ransac_match(self, kp_src, kp_target):
        # input: two groups of keypoints
        # return: homography, binary of inlier/outlier indexes
        kp_src    = np.array(kp_src, dtype=np.float32).reshape(-1,1,2)
        kp_target = np.array(kp_target, dtype=np.float32).reshape(-1,1,2)
        M, S = cv2.findHomography(kp_src, kp_target, cv2.RANSAC, self.tolerate)
        return M, S


    def detect_first_frame(self, frame, bbox_orig):
        # Draw bbox on img
        frame_with_bbox = self.draw_bbox(frame, bbox_orig)
    
        # Detect the keypoints and return them
        #import pdb; pdb.set_trace()
        lt, rt, lb, rb = bbox_orig.reshape(-1, 2)
        frame_crop = frame[lt[0]:rt[0], lt[1]:lb[1]]
        #print('frame_crop: ',frame_crop.shape)
        
        kp_orig, des_orig = self.sift.detectAndCompute(frame_crop, None)
        return frame_with_bbox, kp_orig, des_orig


    def track_detect(self, frame, kp_orig, des_orig, bbox_orig):
        # Detect keypoints
        kp, des = self.sift.detectAndCompute(frame, None)
    
        # Do matching with the previous frame's keypoints
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_orig, des, k=2)     # or using DescriptorMatcher
    
        # Get the matched points
        match_points_pre  = []
        match_points_this = []
    #    for m, n in matches:
    #        img1_idx = m.queryIdx
    #        img2_idx = m.trainIdx
    #        (x1, y1) = kp_orig[img1_idx].pt
    #        (x2, y2) = kp[img2_idx].pt
    #        match_points_pre.append((int(x1), int(y1)))
    #        match_points_this.append((int(x2), int(y2)))

        # Select good points, and get the (x,y) in the img
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        good.sort(key=lambda x: x[0].distance)

        for match in good:
            img1_idx = match[0].queryIdx
            img2_idx = match[0].trainIdx
            (x1, y1) = kp_orig[img1_idx].pt
            (x2, y2) = kp[img2_idx].pt
            match_points_pre.append((int(x1), int(y1)))
            match_points_this.append((int(x2), int(y2)))

        # RANSAC to refine the matched points
        # S is a mask of 1/0 represents matched or mismatched
        M, S = self.ransac_match(match_points_pre, match_points_this)
    
        #x_min, y_min, x_max, y_max = 1000, 1000, 0, 0 
        #S_list = S.ravel().tolist()
        #match_final = []
        #for i in range(len(S_list)):
        #    if S_list[i] == 1:
        #        match_final.append(match_points_this[i])
                #if match_points_this[i][0] < x_min:
                #    x_min = match_points_this[i][0]
                #if match_points_this[i][1] < y_min:
                #    y_min = match_points_this[i][1]
                #if match_points_this[i][0] > x_max:
                #    x_max = match_points_this[i][0]
                #if match_points_this[i][1] > y_max:
                #    y_max = match_points_this[i][1]

        #bbox_tracked = [x_min, y_min, x_max, y_max]
        # Perspect the original bbox
        # bbox_tracked = cv2.warpPerspective(bbox_pre, M)
        bbox_tracked = cv2.perspectiveTransform(np.float32(bbox_orig), M)
        return bbox_tracked


def main(video_input_path, bbox_orig, video_output_path):
    
    video_input = cv2.VideoCapture(video_input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, 30.0, (1280, 720))

    count = 1
    obj_track = Object_tracking(tolerate=5)
    while(True):
        ret, frame = video_input.read()
        #import pdb; pdb.set_trace()
        if ret:
            #import pdb; pdb.set_trace()
            if count == 1:
                #import pdb; pdb.set_trace()
                frame_first, kp_orig, des_orig = obj_track.detect_first_frame(frame, bbox_orig)
                print('i: {0}, bbox: {1}'.format(count, bbox_orig))
                cv2.imshow("Traking Object", frame_first)
                cv2.waitKey(25)
                out.write(frame_first)
                count += 1
            
            else:
                bbox_tracked = obj_track.track_detect(frame, kp_orig, des_orig, bbox_orig)
                frame_tracked = obj_track.draw_bbox(frame, bbox_tracked)   # change bbox shape
                out.write(frame_tracked)
                print('i: {0}, bbox: {1}'.format(count, bbox_tracked))
                count += 1
                cv2.imshow('Traking Object',frame_tracked)
                cv2.waitKey(25)
        else:
             break

    video_input.release()
    out.release()
    cv2.destroyAllWindows()    


if __name__ == '__main__':
    video_input_path  = "./arduino2.mp4"
    video_output_path = "./tracked_arduino.mp4"
    bbox_orig = np.array([[355, 181], [844, 181], [844, 530], [355, 530]]).reshape(-1,1,2)
    #bbox_orig = np.array([[187, 350], [540, 350], [540, 846], [187, 846]]).reshape(-1,1,2)
    main(video_input_path, bbox_orig, video_output_path)
