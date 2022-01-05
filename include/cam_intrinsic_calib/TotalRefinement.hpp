/**
 * @file    TotalRefinement.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
void refineAll(const std::vector<cv::Point2f>& objectPoints,
               const std::vector<std::vector<cv::Point2f>>& imagePointsList, cv::Mat& instrincMat,
               cv::Mat& distortionParams, std::vector<cv::Mat>& extrinsics);
}  // namespace _cv
