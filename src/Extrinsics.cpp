/**
 * @file    Extrinsics.cpp
 *
 * @author  btran
 *
 */

#include <cam_intrinsic_calib/Extrinsics.hpp>

namespace _cv
{
cv::Mat getCameraExtrinsic(const cv::Mat& homography, const cv::Mat& intrinsicMat)
{
    if (homography.size() != cv::Size(3, 3)) {
        throw std::runtime_error("invalid homography size");
    }

    if (intrinsicMat.size() != cv::Size(3, 3)) {
        throw std::runtime_error("invalid camera intrinsic matrix size");
    }

    const cv::Mat& h0 = homography.col(0);
    const cv::Mat& h1 = homography.col(1);
    const cv::Mat& h2 = homography.col(2);

    cv::Mat intrinsicInv = intrinsicMat.inv();
    cv::Mat r0 = intrinsicInv * h0;
    float k = cv::norm(r0);
    r0 /= k;
    cv::Mat r1 = intrinsicInv * h1 / k;
    cv::Mat r2 = r0.cross(r1);

    cv::Mat R = cv::Mat::zeros(3, 3, CV_32FC1);
    r0.copyTo(R.col(0));
    r1.copyTo(R.col(1));
    r2.copyTo(R.col(2));
    R = refineRotationMatrix(R);

    cv::Mat t = intrinsicInv * h2 / k;
    cv::Mat W = cv::Mat::zeros(3, 4, CV_32FC1);
    R.copyTo(W(cv::Rect(0, 0, 3, 3)));
    t.copyTo(W.col(3));

    return W;
}

cv::Mat refineRotationMatrix(const cv::Mat& R)
{
    if (R.size() != cv::Size(3, 3)) {
        throw std::runtime_error("invalid rotation matrix size");
    }

    cv::Mat W, U, Vt;
    cv::SVD::compute(R, W, U, Vt);

    return U * Vt;
}

std::vector<cv::Mat> getCameraExtrinsics(const std::vector<cv::Mat>& homographies, const cv::Mat& intrinsicMat)
{
    int numSamples = homographies.size();
    std::vector<cv::Mat> Rs;
    Rs.reserve(numSamples);
    std::transform(homographies.begin(), homographies.end(), std::back_inserter(Rs),
                   [&intrinsicMat](const auto& elem) { return getCameraExtrinsic(elem, intrinsicMat); });

    return Rs;
}
}  // namespace _cv
