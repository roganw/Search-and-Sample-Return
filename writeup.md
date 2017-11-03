## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./screenshot/preview.png
[image2]: ./screenshot/calibration_images.png
[image3]: ./screenshot/perspect_transform.png
[image4]: ./screenshot/color_thresh.png
[image5]: ./screenshot/rock_detecte.png
[image6]: ./screenshot/coordinate_transform.png 
[image7]: ./screenshot/test_video.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

This is a writeup described how I addressed each point to finish project of Search and Sample Return.

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
Step 1. Prepare test dataset  
First of all, I ran the simulator in manual mode and recorded a series of images.  
Then loaded a random one in jupyter notebook:  
![preview][image1]

Step 2. Load and preview the calibration images  
![calibration_images][image2]

Step 3. Execute perspective transform on calibration image  
I modified the source point to see the difference in perspective transform.
![perspect_transform][image3]

Step 4. Color Thresholding   
When the value of RGB color is larger than `(171, 171, 171)`, it's treated as a pixel of navigable terrain.  
And I used `cv2.bitwise_not` to generate the obstacle threshold image.  
![color_thresh][image4]

Step 5. Rock Detecting  
I defined two method to detect rock samples. The first one is `rock_color_thresh`, which is used to generate a threshold image of rock sample.
The second method is `find_rock`, which is used to check whether it's a rock by using `cv2.findContours` and `cv2.contourArea`.  
I had used `cv2.matchTemplate` before that, but found it was not fast enough in real time.  

When the area of counter(represented by the number of pixels) in rock threshold image is larger than `rock_pixels`, there is a rock in current image.  
```python
#rock threshold will have low and high value to detect it other than path
def rock_color_thresh(img, threshold_low=(86,68,0), threshold_high=(253,210,58)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:, :, 0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:, :, 0] >= threshold_low[0]) & (img[:, :, 0] <= threshold_high[0]) \
                     & (img[:, :, 1] >= threshold_low[1]) & (img[:, :, 1] <= threshold_high[1]) \
                     & (img[:, :, 2] >= threshold_low[2]) & (img[:, :, 2] <= threshold_high[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    return color_select

%time rock_threshed = rock_color_thresh(rock_img)

def find_rock(img, rock_pixels=200):
    matched = False
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours[1]:
        if cv2.contourArea(contour) > rock_pixels:
            matched = True
    return matched

matched = find_rock(rock_threshed)
print(matched)
```
![rock_detecte][image5]

Step 6. Coordinate Transformations  
The method `rover_coords()` was used to convert from image coordinate to rover coordinate.  
And `to_polar_coords()` was used to calculate the distance and angle between the rover and all pixels.
![coordinate_transform][image6]


#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
Step 1. Map into a worldmap  
After previous analysis, I could got the pixels of navigable terrain, obstacles and rock samples easily by these code:
```python
threshed = color_thresh(warped)
obstacle = cv2.bitwise_not(threshed) - 254
rock_thresh = rock_color_thresh(img)
# ...
xpix, ypix = rover_coords(threshed)
obstacle_xpix, obstacle_ypix = rover_coords(obstacle)
if find_rock(rock_thresh):
    rock_xpix, rock_ypix = rover_coords(rock_thresh)
```
Because all of them need to be converted to world coordinate, I defined a method `get_world_xy()` to avoid redundant code:
```python
def get_world_xy(x, y):
    scale = 10
    xpos = data.xpos[data.count]
    ypos = data.ypos[data.count]
    yaw = data.yaw[data.count]
    world_size = data.worldmap.shape[0]
    return pix_to_world(x, y, xpos, ypos, yaw, world_size, scale)
```
Then mapped them into worldmap, and assumed that if a pixel had been judged as navigable, it couldn't be set to obstacle region.
Rock region was always set to white color:
```python
data.worldmap[navigable_y_world, navigable_x_world, 2] = 255
data.worldmap[obstacle_y_world, obstacle_x_world, 0] = 255
already_nav = data.worldmap[:, :, 2] > 0
data.worldmap[already_nav, 0] = 0
# ...
data.worldmap[rock_y_world, rock_x_world, :] = [255, 255, 255]
```

Step 2. Generate test video  
Though that video, I observed the effect and adjusted the parameters in previous method.
![test_video][image7]


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
1.1 Perception  
Referring from the code testing in notebook, most of the work in `perception_step` has been done.  
Additionally, the rover data was get from Rover instance and some logic was added to pitch up rock samples.
When the program find a rock, it not only map the rock to worldmap but also change the rover's navigate direction.
But the rock might missing sometimes, so I added a parameter `Rover.near_rock` and a counter `Rover.missing_rock` to make sure it's not a miscalculation.  
When the rover is near a rock sample but cannot detect it, it'll keep the current angle until `missing_rock > 50`.
```python
rock_appear = find_rock(rock_thresh, rock_pixels=Rover.rock_pixels)
if rock_appear:
    rock_xpix, rock_ypix = rover_coords(rock_thresh)
    rock_x_world, rock_y_world = get_world_xy(Rover, rock_xpix, rock_ypix)
    Rover.worldmap[rock_y_world, rock_x_world, :] = [255, 255, 255]
    Rover.near_rock = True
else:
    Rover.missing_rock += 1
    if Rover.missing_rock > 50:
        Rover.near_rock = False
# 8) Convert rover-centric pixel positions to polar coordinates
# Update Rover pixel distances and angles
    # Rover.nav_dists = rover_centric_pixel_distances
    # Rover.nav_angles = rover_centric_angles
if rock_appear:
    dist, angles = to_polar_coords(rock_xpix, rock_ypix)
    Rover.nav_dists = dist
    Rover.nav_angles = angles
elif not Rover.near_rock:
    dist, angles = to_polar_coords(xpix, ypix)
    Rover.nav_dists = dist
    Rover.nav_angles = angles
```

1.2 Decision  
To make more precise decision, I add several parameter to Rover class. 
`rock_pixels` represent the minimum pixels of a rock sample.  
`missing_rock` is a counter of missing times after detect a rock.  
`near_rock` is a label set when a rock appear in the rover's field of vision.  
`stuck` is a label whether the rover is stuck among the obstacles.  
`stuck_count` and `stuck_times` are used to judge whether the rover is stuck.  
`last_pos` records the last position of rover to calculate the distance between last point and current point.  
```python
self.rock_pixels = 5
self.missing_rock = 0
self.near_rock = False

self.stuck = False
self.stuck_count = 0
self.stuck_times = 100
self.last_pos = None # last position (x, y)

```
After sending the pick up signal, `send_pickup` and `near_rock` are reset to `False`.
```python
if Rover.send_pickup and not Rover.picking_up:
    send_pickup()
    # Reset Rover flags
    Rover.send_pickup = False
    Rover.near_rock = False
```
When the rover is near a rock, I changed the scope of steer angels to `(-30, 30)` to avoid the rock disappear from view.  
And when the mode is `forward`, the velocity is less than 0.01, and the distance from last position also less than 0.01,
the rover might be stuck. When `Rover.stuck_count > Rover.stuck_times`, the rover is stuck definitely. The rover will change to stop mode, and steer in -180 angle.
```python
if Rover.near_rock:
    steer = (-30, 30)
else:
    steer = (-20, 15)
Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), steer[0], steer[1])

if not Rover.picking_up and Rover.vel < 0.01 and Rover.last_pos:
    dist = np.linalg.norm(np.array(Rover.pos) - np.array(Rover.last_pos))
    if dist < 0.01:
        Rover.stuck_count += 1
        if Rover.stuck_count > Rover.stuck_times:
            Rover.stuck = True
            Rover.mode = 'stop'
            Rover.steer = -180
            # Rover.throttle = 1
            Rover.stuck_count = 0
        else:
            Rover.stuck = False
    else:
        Rover.stuck = False
else:
    Rover.stuck = False
```

#### 1. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  
2.1 Environment  
Operation System: MacOS Sierra  
Screen resolution: 800 * 600  
Graphics Quality: Fastest  
Windowed: True  
FPS: 30~40  

2.2 Result  
The rover will pick up 2 to 4 rock samples normally, but might stuck among the obstacles when running around. 

2.3 Improvement  
It's important for the rover to adjust direction to the rock when it appears in the view to pick it up.
But the code need to be improved to avoid go round and round, and how to get out when stuck in obstacles.




