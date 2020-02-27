import cv2
import matplotlib.pyplot as plt
import numpy as np


class Object_tracking(object):
    def __init__(self, ransac_tolerate, bbox_color, bbox_thickness):
        self.sift = cv2.xfeatures2d.SURF_create()

        # parameters for draw bbox
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness

        # parameter for flann
        self.FLANN_INDEX_KDTREE = 0
        self.index_params  = dict(algorithm = self.FLANN_INDEX_KDTREE, trees = 5)
        self.search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

        # parameters for ransac
        self.tolerate = ransac_tolerate

    def draw_bbox(self, img, bbox):
        # Draw a bbox on to a image (in BGR format)
        # return: image with bbox
        img_bbox = cv2.polylines(img, [np.int32(bbox)], True, self.bbox_color, self.bbox_thickness, cv2.LINE_AA)
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
        # import pdb; pdb.set_trace()
        lt, rt, lb, rb = bbox_orig.reshape(-1, 2)
        shift = self.bbox_thickness
        frame_crop = frame[(lt[1]+shift):(lb[1]-shift), (lt[0]+shift):(rt[0]-shift)]
        # cv2.imwrite('frame_crop_1st.jpg', frame_crop)
        #print('frame_crop: ',frame_crop.shape)
        
        kp_orig, des_orig = self.sift.detectAndCompute(frame_crop, None)
        return frame_with_bbox, frame_crop, kp_orig, des_orig


    def track_detect(self, frame, frame_crop_first, kp_orig, des_orig):
        # Detect keypoints
        kp, des = self.sift.detectAndCompute(frame, None)
    
        # Do matching with the previous frame's keypoints
        matches = self.flann.knnMatch(des_orig,des,k=2)
        # bf = cv2.BFMatcher()
        # matches = bf.knnMatch(des_orig, des, k=2)
    
        # Select good points, and get the (x,y) in the img
        good = []
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                good.append([m])
        good.sort(key=lambda x: x[0].distance)

        # Get the matched points
        # match_points_pre  = []
        # match_points_this = []
        # for match in good:
        #     img1_idx = match[0].queryIdx
        #     img2_idx = match[0].trainIdx
        #     (x1, y1) = kp_orig[img1_idx].pt
        #     (x2, y2) = kp[img2_idx].pt
        #     match_points_pre.append((int(x1), int(y1)))
        #     match_points_this.append((int(x2), int(y2)))

        match_points_src = np.float32([ kp_orig[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
        match_points_dst = np.float32([ kp[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)

        # RANSAC to refine the matched points
        # mask is a mask of 1/0 represents matched or mismatched
        # M, mask = self.ransac_match(match_points_src, match_points_dst)
        M, mask = cv2.findHomography(match_points_src, match_points_dst, cv2.RANSAC, self.tolerate)
        matchesMask = mask.ravel().tolist()

        # Perspective transform from the original bbox
        # import pdb; pdb.set_trace()
        h,w = frame_crop_first.shape[0], frame_crop_first.shape[1]
        bbox_first = np.float32([[1,1], [w-1,1], [1,h-1], [w-1,h-1]]).reshape(-1,1,2)
        bbox_tracked = cv2.perspectiveTransform(bbox_first, M)
        return bbox_tracked


def main(video_input_path, bbox_orig, video_output_path):
    
    video_input = cv2.VideoCapture(video_input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, 30.0, (1280, 720))

    count = 1
    obj_track = Object_tracking(ransac_tolerate=3, bbox_color=(0, 255, 255), bbox_thickness=3)
    print('Generating tracked video to {}'.format(video_output_path))
    while(True):
        ret, frame = video_input.read()
        if ret:
            #import pdb; pdb.set_trace()
            if count == 1:
                frame_first, frame_crop_first, kp_orig, des_orig = obj_track.detect_first_frame(frame, bbox_orig)
                # print('i: {0}, bbox: {1}'.format(count, bbox_orig))
                # cv2.imshow("Traking Object", frame_first)
                # cv2.waitKey(25)
                out.write(frame_first)
                count += 1
            
            else:
                bbox_tracked  = obj_track.track_detect(frame, frame_crop_first, kp_orig, des_orig)
                frame_tracked = obj_track.draw_bbox(frame, bbox_tracked)
                out.write(frame_tracked)
                # print('i: {0}, bbox: {1}'.format(count, bbox_tracked))
                # cv2.imshow('Traking Object',frame_tracked)
                # cv2.waitKey(25)
                count += 1
        else:
             break

    video_input.release()
    out.release()
    cv2.destroyAllWindows()    


if __name__ == '__main__':
    video_input_path  = "./arduino.mp4"
    video_output_path = "./tracked_arduino.mp4"
    bbox_orig = np.array([[355, 181], [844, 181], [844, 530], [355, 530]]).reshape(-1,1,2)       # 1280*720
    main(video_input_path, bbox_orig, video_output_path)

    # It will take several minutes to generate video
    # TODO: Resize the image to be smaller
    print('Video has been saved to {}'.format(video_output_path))
