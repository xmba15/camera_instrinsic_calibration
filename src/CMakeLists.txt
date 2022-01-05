cmake_minimum_required(VERSION 3.10)

set(SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/utils/CalibUtils.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/ParamConfig.cpp
  ${PROJECT_SOURCE_DIR}/src/CalibrationHandler.cpp
  ${PROJECT_SOURCE_DIR}/src/Extrinsics.cpp
  ${PROJECT_SOURCE_DIR}/src/Homography.cpp
  ${PROJECT_SOURCE_DIR}/src/LensDistortionModel.cpp
  ${PROJECT_SOURCE_DIR}/src/TotalRefinement.cpp
  ${PROJECT_SOURCE_DIR}/src/Undistortion.cpp
)

add_library(${LIBRARY_NAME}
  SHARED
    ${SOURCE_FILES}
)

target_include_directories(${LIBRARY_NAME}
  SYSTEM PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
  PRIVATE
    ${CERES_INCLUDE_DIRS}
)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    ${OpenCV_LIBRARIES}
  PRIVATE
    ${CERES_LIBRARIES}
)

target_compile_options(${LIBRARY_NAME}
  PRIVATE
     $<$<CONFIG:Debug>:-O0 -g -Wall -Werror>
     $<$<CONFIG:Release>:-O3 -Wall -Werror>
)
