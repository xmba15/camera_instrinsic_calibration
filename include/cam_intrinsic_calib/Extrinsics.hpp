/**
 * @file    Extrinsics.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
cv::Mat getCameraExtrinsic(const cv::Mat& homography, const cv::Mat& intrinsicMat);

cv::Mat refineRotationMatrix(const cv::Mat& R);

std::vector<cv::Mat> getCameraExtrinsics(const std::vector<cv::Mat>& homographies, const cv::Mat& intrinsicMat);
}  // namespace _cv
