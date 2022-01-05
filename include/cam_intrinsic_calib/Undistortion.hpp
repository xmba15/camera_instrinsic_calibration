/**
 * @file    Undistortion.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
/**
 *  \brief find roots of k1 * r^5 + k0 * r^3 +  r - rTilde = 0 for inverting radial model with Newton-Raphson Solver
 *
 */
float fRadInv(float k0, float k1, float rTilde, int numIterations = 20);

cv::Point2f undistortPointNormalizedCoordinate(const cv::Point2f& distorted, const cv::Mat& distortionParams);

cv::Point2f iterativeUndistortPointNormalizedCoordinate(const cv::Point2f& distorted, const cv::Mat& distortionParams,
                                                        int numIteration = 10);

cv::Point2f undistortPoint(const cv::Point2f& sensorCoordDistorted, const cv::Mat& K, const cv::Mat& distortionParams,
                           bool useIterativeMethod = false);

std::vector<cv::Point2f> undistortPoints(const std::vector<cv::Point2f>& sensorCoordDistorteds, const cv::Mat& K,
                                         const cv::Mat& distortionParams, bool useIterativeMethod = false);

cv::Mat getOptimalNewCameraMatrix(const cv::Mat& K, const cv::Mat& distortionParams, const cv::Size& imgSize,
                                  float alpha = 0.);

void initUndistortRectifyMap(const cv::Mat& K, const cv::Mat& distortionParams, const cv::Size& imgSize,
                             const cv::Mat& newCameraMatrix, cv::Mat& map1, cv::Mat& map2);

/**
 *  \brief bilinear interpolation remapping
 *
 */
void remap(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map1, const cv::Mat& map2);
}  // namespace _cv
