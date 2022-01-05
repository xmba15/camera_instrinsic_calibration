/**
 * @file    Utils.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

namespace _cv
{
/**
 *  \brief generate 3d object points on the checkerboard (z = 0)
 *
 *  \param numRow number of rows of checkerboard (inner corner only)
 *  \param numCol number of columns of checkerboard (inner corner only)
 *  \param squareSize checkerboard square size in meters
 *  \return 3d object points
 */
std::vector<cv::Point2f> generateObjectPoints(const int numRow, const int numCol, const float squareSize);

cv::Mat toHomogenous(const cv::Mat& mat);

cv::Mat toInHomogenous(const cv::Mat& mat);
}  // namespace _cv
