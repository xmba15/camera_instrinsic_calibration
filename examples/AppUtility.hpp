/**
 * @file    AppUtility.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <random>

#include <opencv2/viz.hpp>

namespace _cv
{
namespace utils
{
inline std::vector<cv::viz::Color> generateColorCharts(std::uint16_t numSamples = 1000, std::uint16_t seed = 2021)
{
    std::srand(seed);
    std::vector<cv::viz::Color> colors;
    colors.reserve(numSamples);
    for (std::uint16_t i = 0; i < numSamples; ++i) {
        colors.emplace_back(std::rand() % 256, std::rand() % 256, std::rand() % 256);
    }

    return colors;
}
}  // namespace utils
}  // namespace _cv

namespace
{
inline cv::viz::Viz3d initializeWindowWithCamera(cv::Affine3f& camPose, const cv::Matx33d& K, double scale = 0.1)
{
    cv::viz::Viz3d cameraWindow("demo");
    cv::Vec3f camPos(1.0f, 0.0f, 1.0f), camFocalPoint(1.0f, 0.0f, 0.0f), camUpVec(0.f, 1.0f, 0.0f);
    camPose = cv::viz::makeCameraPose(camPos, camFocalPoint, camUpVec);

    cameraWindow.showWidget("camera_coord", cv::viz::WCameraPosition(scale), camPose);
    cameraWindow.showWidget("camera_frustum", cv::viz::WCameraPosition(K, scale), camPose);

    return cameraWindow;
}
}  // namespace
