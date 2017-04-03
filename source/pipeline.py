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
from source.line import Line, DebugInfo


class Pipeline(object):

    def __init__(self, input_video, output_video, debug=False):
        self.input_video = input_video
        self.output_video = output_video
        self.load_camera_calibration()
        self.load_perspective_transform()
        self.left_lane = Line('Left')
        self.right_lane = Line('Right')
        self.detected = False
        self.debug = debug
        self.debug_info = None
        self.debug_img = None

    def sobel(self, image):
        '''
        Return Binary image for lane line detection.  Operates on pre-warped images   
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        s_ch = hls[:, :, 2]
        gray_gradient_x = self.abs_sobel_thresh(gray, thresh=(20, 100))
        s_binary = np.zeros_like(s_ch)
        s_binary[(s_ch >= 120) & (s_ch <= 255)] = 1
        combined = np.zeros_like(gray_gradient_x)
        combined[(gray_gradient_x == 1) | (s_binary == 1)] = 1

        return combined

    def sliding_window_search(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines

        # Edge Protection
        # Exclude this number of pixes from left and right edge of frame
        edge_protection = 100

        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[edge_protection:midpoint])
        rightx_base = np.argmax(histogram[midpoint:-edge_protection]) + midpoint

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

        if self.debug:
            self.debug_info = DebugInfo(nonzerox, nonzeroy, margin, left_lane_inds, right_lane_inds)

        return left_fit, right_fit

    def calculate_radius(self, ploty, left_fitx, right_fitx, height=720):
        # Evaluate Curvature at bottom of frame?
        y_eval = 719
        # Define conversions in x and y from pixels space to meters
        # Warped Image is 720px high and contains approx 5 dashed highway lines
        # Lines are 3m separated by 10m, so 5*(3+10) = 65m
        # Lane width is 500px in my warped image and should be around 3.7m
        ym_per_pix = 73 / 720  # meters per pixel in y dimension
        xm_per_pix = 4.2 / 500  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        return left_curverad, right_curverad

    def sanity_check(self, left_fitx, right_fitx, left_fit, right_fit):
        lane_width = right_fitx - left_fitx
        avg_width = np.average(lane_width)

        abs_n_2_diff = abs(left_fit[0] - right_fit[0])

        return (np.min(lane_width) >= .6 * avg_width and np.max(lane_width) <= 1.3 * avg_width
            and abs_n_2_diff < 0.0005)


    def find_lane_area(self, binary_warped, image, img_size):
        if self.detected:
            left_fit, right_fit = self.margin_search(binary_warped)
        else:
            left_fit, right_fit = self.sliding_window_search(binary_warped)

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        left_curverad, right_curverad = self.calculate_radius(ploty, left_fitx, right_fitx)

        if self.sanity_check(left_fitx, right_fitx, left_fit, right_fit):
            #Store results
            self.detected = True
            self.left_lane.fit = left_fit
            self.right_lane.fit = right_fit
        else:
            left_fit = self.left_lane.fit
            right_fit = self.right_lane.fit
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            left_curverad, right_curverad = self.calculate_radius(ploty, left_fitx, right_fitx)

        if self.debug and self.debug_info:
            self.generate_debug_image(binary_warped, left_fitx, right_fitx, ploty)

        proj = np.zeros_like(image).astype(np.uint8)

        left_points = np.stack((left_fitx, ploty)).astype(np.int).T
        right_points = np.stack((right_fitx, ploty)).astype(np.int).T
        # Maybe stop proj at hood (y > 680)
        points = np.vstack((left_points, np.flipud(right_points)))
        cv2.fillPoly(proj, [points], (0, 255, 0))

        car_perspective = cv2.warpPerspective(proj, self.Minv, img_size, flags=cv2.INTER_LINEAR)
        # Add text and stuff
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        cv2.putText(car_perspective, 'Left Radius: {}m'.format(left_curverad), (10, 50), font, 1.5, color, 3)
        cv2.putText(car_perspective, 'Right Radius: {}m'.format(right_curverad), (10, 100), font, 1.5, color, 3)

        # More testing
        lane_width = right_fitx - left_fitx
        avg_width = np.average(lane_width)
        min_width = np.min(lane_width)
        max_width = np.max(lane_width)
        n_2_diff = left_fit[0] - right_fit[0]
        cv2.putText(car_perspective, 'min width: {}px {}'.format(min_width, min_width/avg_width), (10, 150), font, 1.5, color, 3)
        cv2.putText(car_perspective, 'max width: {}px {}'.format(max_width, max_width/avg_width), (10, 200), font, 1.5, color, 3)
        cv2.putText(car_perspective, 'n^2 diff: {}'.format(n_2_diff), (10, 250), font, 1.5, color, 3)

        return car_perspective

    def process_image(self, image):
        img_size = (image.shape[1], image.shape[0])
        dst = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
        binary = self.sobel(dst)
        binary_warped = cv2.warpPerspective(binary, self.M, img_size, flags=cv2.INTER_LINEAR)
        lane_area_overlay = self.find_lane_area(binary_warped, image, img_size)
        overlay = cv2.addWeighted(image, 1, lane_area_overlay, 0.3, 0)

        if self.debug:
            vis = np.zeros_like(image)

            overlay_resized = cv2.resize(overlay, (853, 480))
            sobel_small = cv2.resize(binary, (426, 240))
            sobel_color = np.dstack((sobel_small, sobel_small, sobel_small)) * 255
            warped_small = cv2.resize(binary_warped, (426, 240))
            warped_color = np.dstack((warped_small, warped_small, warped_small)) * 255

            # vis [h, w]
            vis[240:, :853] = overlay_resized # bottom-left
            vis[:240, :426] = sobel_color
            vis[:240, 426:852] = warped_color

            if self.debug_img is not None:
                debug_small = cv2.resize(self.debug_img, (426, 240))
                vis[:240, 852:1278] = debug_small

            overlay = vis

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

    def load_perspective_transform(self, image_width=1280):
        # These are just constants taken from my IPython notebook
        # a = bottom height
        # b = bottom width (from center)
        # c = top height
        # d = top width (from center)
        a = 680
        b = 378
        c = 480
        d = 88
        w = image_width / 2
        q = 250 # top and bottom width (b, d) for transform
        src = np.float32([[w - b, a], [w - d, c],
                          [w + d, c], [w + b, a]])
        dst = np.float32([[w - q, a], [w - q, c],
                          [w + q, c], [w + q, a]])

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

    def generate_debug_image(self, binary_warped, left_fitx, right_fitx, ploty):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        nonzerox = self.debug_info.nonzerox
        nonzeroy = self.debug_info.nonzeroy
        margin = self.debug_info.margin
        left_lane_inds = self.debug_info.left_lane_inds
        right_lane_inds = self.debug_info.right_lane_inds

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        left_points_plt = np.stack((left_fitx, ploty)).astype(np.int).T
        right_points_plt = np.stack((right_fitx, ploty)).astype(np.int).T
        cv2.polylines(result, [left_points_plt], False, (240, 240, 60), 2)
        cv2.polylines(result, [right_points_plt], False, (240, 240, 60), 2)

        self.debug_img = result


if __name__ == '__main__':
    pipeline = Pipeline('../project_video.mp4', '../output_video.mp4', debug=True)
    pipeline.process()
