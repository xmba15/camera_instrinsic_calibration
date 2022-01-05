/**
 * @file    Utils.cpp
 *
 * @author  btran
 *
 */

#include <numeric>

#include <cam_intrinsic_calib/utils/CalibUtils.hpp>

namespace _cv
{
std::vector<cv::Point2f> generateObjectPoints(const int numRow, const int numCol, const float squareSize)
{
    std::vector<cv::Point2f> targetPoints;
    targetPoints.reserve(numRow * numCol);

    for (int i = 0; i < numRow; ++i) {
        for (int j = 0; j < numCol; ++j) {
            targetPoints.emplace_back(cv::Point2f(j * squareSize, i * squareSize));
        }
    }

    return targetPoints;
}

cv::Mat toHomogenous(const cv::Mat& mat)
{
    if (mat.channels() != 1) {
        throw std::runtime_error("invalid number of channels");
    }

    cv::Mat concatMat;
    cv::hconcat(mat, cv::Mat::ones(mat.rows, 1, mat.type()), concatMat);

    return concatMat;
}

cv::Mat toInHomogenous(const cv::Mat& mat)
{
    if (mat.channels() != 1) {
        throw std::runtime_error("invalid number of channels");
    }

    if (mat.cols < 2) {
        throw std::runtime_error("matrix's number of columns must be at least 2");
    }

    cv::Mat divisor = cv::repeat(mat.col(mat.cols - 1), 1, mat.cols - 1);
    return mat.colRange(0, mat.cols - 1) / divisor;
}
}  // namespace _cv
