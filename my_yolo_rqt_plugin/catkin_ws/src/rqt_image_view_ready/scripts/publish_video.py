#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def publish_video(video_path):
    cap = cv2.VideoCapture(video_path)
    pub = rospy.Publisher('/video_frames', Image, queue_size=10)
    bridge = CvBridge()
    rospy.init_node('video_publisher', anonymous=True)
    rate = rospy.Rate(30)  # 30 FPS 기준

    while not rospy.is_shutdown() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        pub.publish(msg)
        rate.sleep()

    cap.release()

if __name__ == "__main__":
    video_file = "/home/kitech/catkin_ws/SHENZHEN .mp4" # <- 여기에 mp4 경로 입력
    publish_video(video_file)
