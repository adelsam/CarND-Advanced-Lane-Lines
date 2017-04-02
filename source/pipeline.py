import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

# Pipeline for Video Processing
# Load Camera calibration values for distortion correction
# Load constants for Perspective transform
#
# Read Video file
# For each Frame:
# 1. Perspective transform to birds-eye view
# 2. Color/Gradient threshold
# 3. Histogram for lane detection
# 4. Pass through average-r to do sliding window or margin search
# 5. Calculate Lane curvature/distance from center
# 6. Project Lane lines/area back onto input frame
# 7. Write stats out onto input frame
# 8. Assemble frames into output video

class Pipeline(object):

    def __init__(self, input_video, output_video):
        self.input_video = input_video
        self.output_video = output_video
        self.load_camera_calibration()
        self.load_perspective_transform()

    def process_image(self, image):
        img_size = (image.shape[1], image.shape[0])
        dst = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        warped = cv2.warpPerspective(dst, self.M, img_size, flags=cv2.INTER_LINEAR)
        # sobel = abs_sobel_thresh(dst)

        return warped

    def process(self):
        print('Reading Video file {}'.format(self.input_video))
        clip1 = VideoFileClip(self.input_video)
        white_clip = clip1.fl_image(self.process_image)
        white_clip.write_videofile(self.output_video, audio=False)
        print('Wrote Output Video file {}'.format(self.output_video))

    def load_camera_calibration(self):
        with open("../camera_cal/camera_dist_pickle.p", "rb") as f:
            dist_pickle = pickle.load(f)
            self.mtx = dist_pickle["mtx"]
            self.dist = dist_pickle["dist"]

    def load_perspective_transform(self):
        # These are just constants taken from my IPython notebook
        src = np.float32([[262, 680], [550, 480], [730, 480], [1040, 680]])
        dst = np.float32([[400, 680], [400, 400], [850, 400], [850, 680]])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)


if __name__ == '__main__':
    pipeline = Pipeline('../project_video.mp4', '../output_video.mp4')
    pipeline.process()
