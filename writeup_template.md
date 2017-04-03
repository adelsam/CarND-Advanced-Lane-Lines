## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/test_undist.jpg "Undistorted"
[image2]: ./output_images/undist_road.png "Undistorted"
[image3]: ./output_images/sobel_dual.png "Sobel Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/sliding_window.png "Sliding Window Search"
[image6]: ./output_images/warped_area.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I calibrated the camera in the third code cell of the IPython notebook located in "./Lane Finding.ipynb".  

I basically used the code from the lesson on calibration, reading in all 20 calibration images, using cv2.findChessboardCorners to find the corners in the supplied images.  Once the calibrated, I can correct for camera distortion in images using the `cv2.undistort()` function, for example: 

![alt text][image1]

I stored the calibration information (`mtx`, `dist`) in a pickle file so that it would be easy to load later.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
My single-image pipeline starts in the 8th cell of the IPython notebook.  I built the workflow one step at a time, using the interactive notebook to validate results as I went.  The first step is to undistort the camera images for example:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
Since this assignment is primarily a computer vision asignment, I calculated a variety of color transforms and thresholds to try and extract the lane line pixels from sample images.  These functions are in the 7th IPython cell (i.e. `dev abs_sobel_thresh`).  The `conf_sobel` function was used to combine a sobel threshold on the gray channel with a simple threshold on the s channel.  I used the np.dstack() technique to visualize the pixels returned by each of these filters prior to combining them into a single binary image (Cell 12: `sobel_combined`).

Here's an example of the two different filters being used on a sample image:
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In order to transform images from the camera perspective to a top-down "bird's eye" perspective, I chose sample points centered on the horizontal center of the image that made a sample image of straight lines appear roughly parallel.  This code is in the 10th cell of the IPython notebook.  

Here is the example image used to help select the points for the transform:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Using the code from the lesson, I implemented the sliding window search using a histogram of the bottom half of the warped binary sobel image to locate the lane lines in the first frame.  There was some noise on the sides of some of my images, so I just excluded 100 px from both sides from evaluation.  The histogram is visualized in the 13th cell of the notebook and the sliding window algorithm follows in the 14th cell.  A visualization of this algorithm is generated in Cell 15:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
I calculated the lane curvature in Cell 30, first scaling the pixel measurements into approximate meters and then re-calculating the polynomial best fit line in the appropriate units.  I dealt with the lane-center measurement in the video pipeline code by comparing the theoritical center of the frame (640px) with the point half-way between the two lane lines.  For this, see `calculate_offset` in `pipeline.py`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

To plot the detected lane area back down onto the road image taken by the camera, I first created top-down view of the lane using `cv2.fillPoly` with the set of points calculated by the polynomial fit.  This code is in the 31st cell of the notebook.  I found it was easier to warp the region generated from the cv2 library rather than plotting the points using matplotlib.  Once I had the top down image, I used `cv2.warpPerspective` with the inverse matrix calculated earlier, and `cv2.addWeighted` to combine the warped lane area with the original image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video.mp4)

To process the video, I copied most of the code described above into a file `./source/pipeline.py` and completed the asignment in an IDE.

#### 2. Sanity Checks
To help process the video, I added sanity checking and averaging to my lane line calculation.  Since the warped image is a top down view, we expect the lane lines to be parallel.  I verified that the distance between the two lane lines was within a small percentage of the average based on the calculated lines and discarded any calculated best fit lines that failed that test.

We also expect the curvature of highway lane lines to be limited.  I approximated validating the curvature of the lane by checking the magnitude of the first coeffecient of the polynomial function, discarding any above a constant value.

In addition to this filtering, I also stored a queue of the last 5 reasonable fit lines, and just projected an average of these fit lines over frames that failed the sanity checks.  This provided reasonable results and the lane detection algorithm usually recovered quickly.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced some issues finding good values for the initial warp distortion.  Since the camera was not centered on the lane, I initially chose values that attempted to compensate for the offset, which introduced a lot of additional distortion in my images.  I eventually went back and centered all the points used for the transform horizontally and got much more perdictable results.

I also struggled a little to troubleshoot frames that were causing the detection to fail.  To better encapsulate the lane line search, I created a `margin_search` function (cell 22) and returned all the results in a single `Result` object (Cell 16).  This made it much easier to test and I wish that I had done so earlier.

Finally, in my video-processing `pipeline.py`, I added an option to create a debug video that output each of the steps of the pipeline along with the final overlay image in the output video.  This made it much easier to track how the algorithm was performing.

