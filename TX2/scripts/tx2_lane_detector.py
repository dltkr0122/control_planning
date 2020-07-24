#!/usr/bin/env python
 
import rospy
import cv2
import numpy as np
import os, rospkg
import json

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridgeError

from utils import BEVTransform, CURVEFit, draw_lane_img

class IMGParser:
    def __init__(self):

        self.img = None
        self.set_cam(1)

    def set_cam(self, _index):
        self.cam = cv2.VideoCapture(int(_index))

    def get_image(self):
        ret, img = self.cam.read()
        return ret, img

    def get_bi_img(self):
        ret, img_bgr = self.get_image()

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        lower_wlane = np.array([75,0,220])
        upper_wlane = np.array([175,20,255])
        img_wlane = cv2.inRange(img_hsv, lower_wlane, upper_wlane)

        # img_wlane = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        return img_wlane


if __name__ == '__main__':
    
    rp = rospkg.RosPack()
    
    currentPath = rp.get_path("traffic_example")
    
    with open(os.path.join(currentPath, 'sensor/sensor_params.json'), 'r') as fp:
        sensor_params = json.load(fp)

    params_cam = sensor_params["params_cam"]

    rospy.init_node('lane_detector', anonymous=True)

    image_parser = IMGParser()
    bev_op = BEVTransform(params_cam=params_cam)
    curve_learner = CURVEFit(order=1)

    rate = rospy.Rate(20)

    while not rospy.is_shutdown():

        if image_parser.edges is not None:

            img_warp = bev_op.warp_bev_img(image_parser.edges)
            lane_pts = bev_op.recon_lane_pts(image_parser.edges)

            x_pred, y_pred_l, y_pred_r = curve_learner.fit_curve(lane_pts)

            curve_learner.write_path_msg(x_pred, y_pred_l, y_pred_r)

            curve_learner.pub_path_msg()

            xyl, xyr = bev_op.project_lane2img(x_pred, y_pred_l, y_pred_r)

            img_warp1 = draw_lane_img(img_warp, xyl[:, 0].astype(np.int32),
                                                xyl[:, 1].astype(np.int32),
                                                xyr[:, 0].astype(np.int32),
                                                xyr[:, 1].astype(np.int32))
                                                
            cv2.imshow("Image window", img_warp1)
            cv2.waitKey(1)

            rate.sleep()
