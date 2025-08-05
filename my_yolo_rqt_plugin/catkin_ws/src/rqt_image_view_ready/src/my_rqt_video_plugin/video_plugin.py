#!/usr/bin/env python3
import os
import rospy
import cv2
from python_qt_binding.QtWidgets import QWidget, QLabel, QPushButton
from python_qt_binding.QtCore import QTimer, Qt
from python_qt_binding import loadUi
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rqt_gui_py.plugin import Plugin

class VideoPlugin(Plugin):

    def __init__(self, context):
        super(VideoPlugin, self).__init__(context)
        self.setObjectName('MyPlugin')
        
        self._widget = QWidget()
        ui_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '/home/kitech/catkin_ws/src/rqt_image_view_ready/resource/streaming.ui')
        loadUi(ui_file, self._widget)
        if context.serial_number() > 1:
            self._widget.setWindowTitle(f'{self._widget.windowTitle()} ({context.serial_number()})')

        context.add_widget(self._widget)

        self.bridge = CvBridge()
        self.play_button = self._widget.findChild(QPushButton, 'play_button')
        self.video_label = self._widget.findChild(QLabel, 'video_label')

        self.play_button.clicked.connect(self.start_video)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.subscriber = rospy.Subscriber('/video_frames', Image, self.image_callback)
        
    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape

            qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_img))
        except Exception as e:
            rospy.logerr(f"Image conversion error: {e}")
    def start_video(self):
        video_path = "/home/kitech/catkin_ws/src/rqt_image_view_ready/SHENZHEN.mp4"
        if not os.path.exists(video_path):
            rospy.logerr(f"Video file not found: {video_path}")
            return
        self.cap = cv2.VideoCapture(video_path)
        self.timer.start(30)  # 대략 30FPS

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                from PyQt5.QtGui import QImage, QPixmap
                qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_img))
            else:
                self.timer.stop()
                self.cap.release()
