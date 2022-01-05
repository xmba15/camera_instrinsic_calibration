/**
 * @file    LensDistortionModel.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
cv::Mat estimateRadialLensDistortion(const std::vector<cv::Point2f>& objectPoints,
                                     const std::vector<std::vector<cv::Point2f>>& imagePointsList,
                                     const cv::Mat& intrinsicMat, const std::vector<cv::Mat>& extrinsics);
}  // namespace _cv
