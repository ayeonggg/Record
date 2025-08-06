#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def publish_video(video_path, pub, bridge, rate):
    while not rospy.is_shutdown():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            rospy.logerr(f"Cannot open video file: {video_path}")
            break

        while not rospy.is_shutdown() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            pub.publish(msg)
            rate.sleep()

        cap.release()
        rospy.loginfo(f"Finished playing {video_path}, restarting...")

if __name__ == "__main__":
    video_files = [
        "/home/ayeong/my_yolo_rqt_plugin/catkin_ws/src/rqt_image_view_ready/SHENZHEN.mp4",
        "/home/ayeong/my_yolo_rqt_plugin/catkin_ws/xdz.mp4"
    ]

    rospy.init_node('video_publisher', anonymous=True)
    pub = rospy.Publisher('/video_frames', Image, queue_size=10)
    bridge = CvBridge()
    rate = rospy.Rate(30)  # 30 FPS 기준

    # 영상들을 무한 반복 재생
    while not rospy.is_shutdown():
        for video_file in video_files:
            rospy.loginfo(f"Publishing video: {video_file}")
            publish_video(video_file, pub, bridge, rate)
            if rospy.is_shutdown():
                break

    rospy.loginfo("All videos published.")
