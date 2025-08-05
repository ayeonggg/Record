cmake_minimum_required(VERSION 3.0.2)
project(rqt_image_view_ready)

find_package(catkin REQUIRED COMPONENTS
  rospy
  rqt_gui
  rqt_gui_py
  sensor_msgs
  cv_bridge
)

catkin_python_setup()

catkin_package()

install(DIRECTORY resource
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES
  plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
