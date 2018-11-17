#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial import KDTree
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint

import math
import numpy as np
import tf

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.waypoints_2d = None
        self.pose = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        rospy.Timer(rospy.Duration(1.0 / 50.0), lambda event: self.tick())

        rospy.spin()

    def tick(self):
        if self.pose and self.base_waypoints and self.waypoint_tree:
            self.final_waypoints_pub.publish(self.generate_lane())

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = 2 * MAX_DECEL * dist
            if vel < 1.0:
                vel = 0.0

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def get_direction(self):
        orientation = self.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion([orientation.x,
                                                         orientation.y,
                                                         orientation.z,
                                                         orientation.w])
        yaw = euler[2]
        pos = self.pose.pose.position  
        cur_pos_2d = np.array([pos.x, pos.y])
        next_pos_2d = cur_pos_2d + np.array([np.cos(yaw), np.sin(yaw)]) 
        cur_closest_idx = self.waypoint_tree.query(cur_pos_2d, 1)[1]
        prev_wpt = np.array(self.waypoints_2d[cur_closest_idx - 1])
        if np.linalg.norm(cur_pos_2d - prev_wpt) < np.linalg.norm(next_pos_2d - prev_wpt):
            return 1
        else:
            return -1

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        num_waypts = len(self.waypoints_2d)
        closest_coord = self.waypoints_2d[closest_idx]
        prev_idx = (closest_idx + (-1 if self.get_direction() > 0 else 1)) % num_waypts
        prev_coord = self.waypoints_2d[prev_idx]

        a = np.array(prev_coord)
        b = np.array(closest_coord)
        c = np.array([x, y])

        dot_prod = np.dot(b - a, c - b)

        if dot_prod > 0:
            return (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx


    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
