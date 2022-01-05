/**
 * @file    Homography.cpp
 *
 * @author  btran
 *
 */

#include <algorithm>
#include <numeric>
#include <utility>

#include <Eigen/Core>
#include <ceres/ceres.h>

#include <cam_intrinsic_calib/Homography.hpp>
#include <cam_intrinsic_calib/utils/utils.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>

namespace
{
std::pair<float, float> calculateMeanVariance(const std::vector<float>& xs)
{
    if (xs.empty()) {
        throw std::runtime_error("empty vector");
    }

    const float mean = std::accumulate(std::begin(xs), std::end(xs), 0.0) / xs.size();
    const float variance = std::accumulate(std::begin(xs), std::end(xs), 0.0,
                                           [mean](float sum, const float elem) {
                                               const float tmp = elem - mean;
                                               return sum + tmp * tmp;
                                           }) /
                           xs.size();
    return std::make_pair(mean, variance);
}

struct RefineHomographyResidual {
    RefineHomographyResidual(const cv::Point2f& objectPoint, const cv::Point2f& imagePoint)
        : m_objectPoint(objectPoint)
        , m_imagePoint(imagePoint)
    {
    }

    template <typename T> bool operator()(const T* const coeffs, T* residuals) const
    {
        Eigen::Matrix<T, 3, 3> coeffsMat(coeffs);
        coeffsMat.transposeInPlace();
        Eigen::Matrix<T, 2, 1> objectPointsMat(static_cast<T>(m_objectPoint.x), static_cast<T>(m_objectPoint.y));
        Eigen::Matrix<T, 2, 1> imagePointsMat(static_cast<T>(m_imagePoint.x), static_cast<T>(m_imagePoint.y));

        Eigen::Matrix<T, 2, 1> diff = imagePointsMat - (coeffsMat * objectPointsMat.homogeneous()).hnormalized();
        Eigen::Matrix<T, 2, 1> invDiff =
            objectPointsMat - (coeffsMat.inverse() * imagePointsMat.homogeneous()).hnormalized();

        residuals[0] = diff[0];
        residuals[1] = diff[1];
        residuals[2] = invDiff[0];
        residuals[3] = invDiff[1];
        return true;
    }

 private:
    const cv::Point2f& m_objectPoint;
    const cv::Point2f& m_imagePoint;
};
}  // namespace

namespace _cv
{
cv::Mat calculateNormalizationMatrix(const std::vector<cv::Point2f>& points)
{
    if (points.empty()) {
        throw std::runtime_error("empty points");
    }

    std::vector<float> xs, ys;
    xs.reserve(points.size());
    ys.reserve(points.size());
    for (const auto& point : points) {
        xs.emplace_back(point.x);
        ys.emplace_back(point.y);
    }

    float xsMean, xsVariance, ysMean, ysVariance;
    std::tie(xsMean, xsVariance) = ::calculateMeanVariance(xs);
    std::tie(ysMean, ysVariance) = ::calculateMeanVariance(ys);

    float sx = std::sqrt(2 / (xsVariance + std::numeric_limits<float>::epsilon()));
    float sy = std::sqrt(2 / (ysVariance + std::numeric_limits<float>::epsilon()));

    // clang-format off
    cv::Mat normMatrix = (cv::Mat_<float>(3, 3) << sx , 0.0, -sx * xsMean,
                                                   0.0, sy , -sy * ysMean,
                                                   0.0, 0.0, 1.0
                         );
    // clang-format on

    return normMatrix;
}

cv::Mat calculateHomography(const std::vector<cv::Point2f>& objectPoints, const std::vector<cv::Point2f>& imagePoints)
{
    int numImagePoints = imagePoints.size();

    cv::Mat objectPointsMat(numImagePoints, 2, CV_32FC1);
    cv::Mat imagePointsMat(numImagePoints, 2, CV_32FC1);
    std::memcpy(objectPointsMat.data, objectPoints.data(), objectPointsMat.total() * objectPointsMat.elemSize());
    std::memcpy(imagePointsMat.data, imagePoints.data(), imagePointsMat.total() * imagePointsMat.elemSize());

    // data normalization
    cv::Mat objectPointsNormMat = calculateNormalizationMatrix(objectPoints);
    cv::Mat imagePointsNormMat = calculateNormalizationMatrix(imagePoints);
    cv::Mat normalizedObjectPoints = toInHomogenous((objectPointsNormMat * toHomogenous(objectPointsMat).t()).t());
    cv::Mat normalizedImagePoints = toInHomogenous((imagePointsNormMat * toHomogenous(imagePointsMat).t()).t());

    cv::Mat M = cv::Mat::zeros(2 * numImagePoints, 9, CV_32FC1);

    for (int i = 0; i < numImagePoints; ++i) {
        const float* curObjectPoint = normalizedObjectPoints.ptr<float>(i);
        const float* curImagePoint = normalizedImagePoints.ptr<float>(i);

        cv::Mat firstRow =
            (cv::Mat_<float>(1, 9) << -curObjectPoint[0], -curObjectPoint[1], -1, 0, 0, 0,
             curImagePoint[0] * curObjectPoint[0], curImagePoint[0] * curObjectPoint[1], curImagePoint[0]);
        firstRow.copyTo(M.row(2 * i));

        cv::Mat secondRow =
            (cv::Mat_<float>(1, 9) << 0, 0, 0, -curObjectPoint[0], -curObjectPoint[1], -1,
             curImagePoint[1] * curObjectPoint[0], curImagePoint[1] * curObjectPoint[1], curImagePoint[1]);
        secondRow.copyTo(M.row(2 * i + 1));
    }
    cv::Mat W, U, Vt;
    cv::SVD::compute(M, W, U, Vt);

    cv::Mat normalizedH = Vt.row(8).reshape(1, 3);
    cv::Mat H = imagePointsNormMat.inv() * normalizedH * objectPointsNormMat;

    H = refineHomography(objectPoints, imagePoints, H);

    return H;
}

cv::Mat refineHomography(const std::vector<cv::Point2f>& objectPoints, const std::vector<cv::Point2f>& imagePoints,
                         const cv::Mat& initialGuessH)
{
    int numImagePoints = imagePoints.size();
    cv::Mat refinedH;
    initialGuessH.convertTo(refinedH, CV_64FC1);

    ceres::Problem prob;
    for (int i = 0; i < numImagePoints; ++i) {
        prob.AddResidualBlock(new ceres::AutoDiffCostFunction<RefineHomographyResidual, 4, 9>(
                                  new RefineHomographyResidual(objectPoints[i], imagePoints[i])),
                              nullptr, refinedH.ptr<double>());
    }
    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &prob, &summary);
    refinedH /= refinedH.ptr<double>(2)[2];
    refinedH.convertTo(refinedH, CV_32FC1);

    return refinedH;
}

std::vector<cv::Mat> calculateHomographies(const std::vector<cv::Point2f>& objectPoints,
                                           const std::vector<std::vector<cv::Point2f>>& imagePointsList)
{
    int numSamples = imagePointsList.size();
    std::vector<cv::Mat> Hs;
    Hs.reserve(numSamples);
    std::transform(imagePointsList.begin(), imagePointsList.end(), std::back_inserter(Hs),
                   [&objectPoints](const auto& elem) { return calculateHomography(objectPoints, elem); });
    return Hs;
}

cv::Mat getCameraIntrinsicFromHomographies(const std::vector<cv::Mat>& homographies)
{
    int numSamples = homographies.size();

    for (const auto& homography : homographies) {
        if (homography.size() != cv::Size(3, 3)) {
            throw std::runtime_error("invalid size of homography matrix");
        }
    }

    auto getVij = [](const cv::Mat& h, int i, int j) {
        cv::Mat vij = cv::Mat::zeros(1, 6, CV_32FC1);
        float* data = vij.ptr<float>(0);
        data[0] = h.ptr<float>(0)[i] * h.ptr<float>(0)[j];
        data[1] = h.ptr<float>(0)[i] * h.ptr<float>(1)[j] + h.ptr<float>(1)[i] * h.ptr<float>(0)[j];
        data[2] = h.ptr<float>(1)[i] * h.ptr<float>(1)[j];
        data[3] = h.ptr<float>(2)[i] * h.ptr<float>(0)[j] + h.ptr<float>(0)[i] * h.ptr<float>(2)[j];
        data[4] = h.ptr<float>(2)[i] * h.ptr<float>(1)[j] + h.ptr<float>(1)[i] * h.ptr<float>(2)[j];
        data[5] = h.ptr<float>(2)[i] * h.ptr<float>(2)[j];

        return vij;
    };

    cv::Mat V = cv::Mat::zeros(2 * numSamples, 6, CV_32FC1);
    for (int i = 0; i < numSamples; ++i) {
        getVij(homographies[i], 0, 1).copyTo(V.row(2 * i));
        cv::Mat(getVij(homographies[i], 0, 0) - getVij(homographies[i], 1, 1)).copyTo(V.row(2 * i + 1));
    }
    cv::Mat W, U, Vt;
    cv::SVD::compute(V, W, U, Vt);

    cv::Mat b = Vt.row(5);
    const float* bData = b.ptr<float>();

    float w = bData[0] * bData[2] * bData[5] - bData[1] * bData[1] * bData[5] - bData[0] * bData[4] * bData[4] +
              2 * bData[1] * bData[3] * bData[4] - bData[2] * bData[3] * bData[3];
    float d = bData[0] * bData[2] - bData[1] * bData[1];
    float alpha = std::sqrt(w / (d * bData[0]));
    float beta = std::sqrt(w / (d * d) * bData[0]);
    float gamma = std::sqrt(w / (d * d * bData[0])) * bData[1];
    float uc = (bData[1] * bData[4] - bData[2] * bData[3]) / d;
    float vc = (bData[1] * bData[3] - bData[0] * bData[4]) / d;

    // clang-format off
    cv::Mat K = (cv::Mat_<float>(3, 3) << alpha, gamma, uc,
                                          0, beta, vc,
                                          0, 0, 1
                );
    // clang-format on

    return K;
}
}  // namespace _cv
