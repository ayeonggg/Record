#!/usr/bin/env python3
import os
import rospy
from python_qt_binding.QtWidgets import QWidget, QLabel, QPushButton
from python_qt_binding import loadUi
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rqt_gui_py.plugin import Plugin

class VideoPlugin(Plugin):
    def __init__(self, context):
        super(VideoPlugin, self).__init__(context)
        self.setObjectName('MyPlugin')

        # Load UI
        self._widget = QWidget()
        ui_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               '/home/ayeong/my_yolo_rqt_plugin/catkin_ws/src/rqt_image_view_ready/resource/streaming.ui')
        loadUi(ui_file, self._widget)
        context.add_widget(self._widget)

        self.bridge = CvBridge()
        self.sub_rgb = None
        self.sub_meta = None

        # Find widgets
        self.button_rgb = self._widget.findChild(QPushButton, 'pushButton')
        self.button_meta = self._widget.findChild(QPushButton, 'pushButton_2')
        self.label_rgb = QLabel()
        self.label_meta = QLabel()

        layout_rgb = self._widget.findChild(QWidget, 'verticalLayoutWidget').layout()
        layout_meta = self._widget.findChild(QWidget, 'verticalLayoutWidget_2').layout()

        self.label_rgb.setFixedSize(320, 240)
        self.label_meta.setFixedSize(320, 240)
        self.label_rgb.setStyleSheet("background-color: black")
        self.label_meta.setStyleSheet("background-color: black")

        layout_rgb.addWidget(self.label_rgb)
        layout_meta.addWidget(self.label_meta)

        self.button_rgb.clicked.connect(self.subscribe_rgb)
        self.button_meta.clicked.connect(self.subscribe_meta)

    def subscribe_rgb(self):
        if not self.sub_rgb:
            self.sub_rgb = rospy.Subscriber("/yolov11/segmentation", Image, self.cb_rgb)

    def subscribe_meta(self):
        if not self.sub_meta:
            self.sub_meta = rospy.Subscriber("/yolov11/segmentation", Image, self.cb_meta)

    def cb_rgb(self, msg):
        self.update_image(msg, self.label_rgb)

    def cb_meta(self, msg):
        self.update_image(msg, self.label_meta)

    def update_image(self, msg, label):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qimg).scaled(
                label.width(), label.height(), Qt.KeepAspectRatio))
        except Exception as e:
            rospy.logerr(f"Image update error: {e}")
