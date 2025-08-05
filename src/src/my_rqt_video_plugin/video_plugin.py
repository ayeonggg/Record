import os
import rospy
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget, QLabel
from rqt_gui_py.plugin import Plugin
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PyQt5.QtGui import QImage, QPixmap

class VideoPlugin(Plugin):
    def __init__(self, context):
        super(VideoPlugin, self).__init__(context)
        self.setObjectName('VideoPlugin')
        self._widget = QWidget()

        # UI 파일 경로
        ui_file = os.path.join(os.path.dirname(__file__), '/home/kitech/catkin_ws/src/resource/streaming.ui')
        loadUi(os.path.abspath(ui_file), self._widget)
        self._widget.setObjectName('VideoPluginUi')

        # QLabel 객체 찾기
        self.image_label = self._widget.findChild(QLabel, 'video_label')
        context.add_widget(self._widget)

        # ROS 초기화 생략 (RQt 내부에서 처리됨)
        self.bridge = CvBridge()

        # 이미지 토픽 구독 (여기선 /video_frames 사용)
        rospy.Subscriber("/video_frames", Image, self.image_callback)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        qimg = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)