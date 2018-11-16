# Capstone project
Objective of the project is to ensure that Carla drives itself over a predefined track respecting traffic signals. This is a project where different parts of the stack that has been developed over the course comes together.

## Architecture
The system is implemented using ROS. Parts of the stack that were expected to be built includes:

### Waypoint updater
This node is responsible for generating waypoints such that the vehicle is always in the center of the lane. Waypoint information is available from configuration. Stopline waypoints are also fed in to this node and this node has the responsibility to generate waypoints for stopping at stoplines. 

### Traffic light detection
A deep neural network based on single shot detection is used to perform traffic light detection. Images from sim and recorded bag file were run through tensorflow object detection api to identify bounding boxes. A custom object detection pipeline was then built on top SSD model to classify red, yellow and green traffic lights. SSD provided the best trade-off between speed and accuracy. Faster rcnn and RFCN were also experimented with but was much slower. SSD provided averaged about 50 frames per second.

### DBW Steering and throttle controller
Drive By Wire is already available thought Dataspeed DBW. Throttle controller is a PID controller and yaw/steering angle is computed from linear and angular velocities.

## Group info
Rohith Menon - rohithmenon@gmail.com
