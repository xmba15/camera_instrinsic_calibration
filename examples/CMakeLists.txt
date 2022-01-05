cmake_minimum_required(VERSION 3.10)

add_executable(calib_app
  ${CMAKE_CURRENT_LIST_DIR}/CalibrationApp.cpp
)

target_link_libraries(calib_app
  PUBLIC
    ${LIBRARY_NAME}
)

add_executable(undistortion_app
  ${CMAKE_CURRENT_LIST_DIR}/UndistortionApp.cpp
)

target_link_libraries(undistortion_app
  PUBLIC
    ${LIBRARY_NAME}
)
