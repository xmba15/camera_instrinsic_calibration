/**
 * @file    Homography.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

namespace _cv
{
cv::Mat calculateNormalizationMatrix(const std::vector<cv::Point2f>& points);

cv::Mat calculateHomography(const std::vector<cv::Point2f>& objectPoints, const std::vector<cv::Point2f>& imagePoints);

cv::Mat refineHomography(const std::vector<cv::Point2f>& objectPoints, const std::vector<cv::Point2f>& imagePoints,
                         const cv::Mat& initialGuessH);

std::vector<cv::Mat> calculateHomographies(const std::vector<cv::Point2f>& objectPoints,
                                           const std::vector<std::vector<cv::Point2f>>& imagePointsList);

cv::Mat getCameraIntrinsicFromHomographies(const std::vector<cv::Mat>& homographies);
}  // namespace _cv
