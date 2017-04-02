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
from source.line import Line


class Pipeline(object):

    def __init__(self, input_video, output_video):
        self.input_video = input_video
        self.output_video = output_video
        self.load_camera_calibration()
        self.load_perspective_transform()
        self.left_lane = Line('Left')
        self.right_lane = Line('Right')

    def sobel(self, image):
        '''
        Return Binary image for lane line detection.  Operates on pre-warped images   
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_ch = hls[:, :, 2]
        gray_gradient_x = self.abs_sobel_thresh(gray, thresh=(20, 100))
        s_ch_gradient_x = self.abs_sobel_thresh(s_ch, thresh=(20, 100))
        mag_and_dir_thresh = self.mag_and_dir_thresh(gray, mag_thresh=(30, 100),
                                                     dir_thresh=(0.7, 1.3))
        combined = np.zeros_like(gray_gradient_x)
        combined[(gray_gradient_x == 1) | (s_ch_gradient_x == 1) |
                 (mag_and_dir_thresh == 1)] = 1

        return combined

    def sliding_window_search(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit

    def margin_search(self, binary_warped):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        left_fit = self.left_lane.fit
        right_fit = self.right_lane.fit

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
        (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit

    def find_lane_area(self, binary_warped, image, img_size):
        if self.left_lane.detected and self.right_lane.detected:
            left_fit, right_fit = self.margin_search(binary_warped)
        else:
            left_fit, right_fit = self.sliding_window_search(binary_warped)

        #Store results
        self.left_lane.detected = True
        self.left_lane.fit = left_fit
        self.right_lane.detected = True
        self.right_lane.fit = right_fit

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        proj = np.zeros_like(image).astype(np.uint8)

        left_points = np.stack((left_fitx, ploty)).astype(np.int).T
        right_points = np.stack((right_fitx, ploty)).astype(np.int).T
        # Maybe stop proj at hood (y > 680)
        points = np.vstack((left_points, np.flipud(right_points)))
        cv2.fillPoly(proj, [points], (0, 255, 0))

        return cv2.warpPerspective(proj, self.Minv, img_size, flags=cv2.INTER_LINEAR)


    def process_image(self, image):
        img_size = (image.shape[1], image.shape[0])
        dst = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        binary = self.sobel(dst)
        binary_warped = cv2.warpPerspective(binary, self.M, img_size, flags=cv2.INTER_LINEAR)
        lane_area_overlay = self.find_lane_area(binary_warped, image, img_size)
        overlay = cv2.addWeighted(image, 1, lane_area_overlay, 0.3, 0)

        return overlay

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

    def abs_sobel_thresh(self, gray, thresh, orient='x', sobel_kernel=3):
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return the result
        return binary_output

    def mag_and_dir_thresh(self, gray, mag_thresh, dir_thresh, sobel_kernel=3):
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))
        # 5) Create a binary mask where mag thresholds are met
        mag_output = np.zeros_like(scaled_sobel)
        mag_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        grad_dir = np.arctan2(abs_sobely, abs_sobelx)
        dir_output = np.zeros_like(grad_dir)
        dir_output[(grad_dir >= dir_thresh[0]) & (grad_dir <= dir_thresh[1])] = 1

        mag_and_dir = np.zeros_like(scaled_sobel)
        mag_and_dir[((mag_output == 1) & (dir_output == 1))]
        # Return the result
        return mag_and_dir


if __name__ == '__main__':
    pipeline = Pipeline('../project_video.mp4', '../output_video.mp4')
    pipeline.process()
