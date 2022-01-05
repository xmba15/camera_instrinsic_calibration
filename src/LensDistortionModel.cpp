/**
 * @file    LensDistortionModel.cpp
 *
 * @author  btran
 *
 */

#include <cam_intrinsic_calib/LensDistortionModel.hpp>
#include <cam_intrinsic_calib/utils/utils.hpp>

namespace _cv
{
cv::Mat estimateRadialLensDistortion(const std::vector<cv::Point2f>& objectPoints,
                                     const std::vector<std::vector<cv::Point2f>>& imagePointsList,
                                     const cv::Mat& intrinsicMat, const std::vector<cv::Mat>& extrinsics)
{
    int numSamples = extrinsics.size();
    int numPointsEachSample = objectPoints.size();

    cv::Mat D = cv::Mat::zeros(2 * numSamples * numPointsEachSample, 2, CV_32FC1);
    cv::Mat d = cv::Mat::zeros(2 * numSamples * numPointsEachSample, 1, CV_32FC1);

    cv::Point2f principalPoint(intrinsicMat.ptr<float>(0)[2], intrinsicMat.ptr<float>(1)[2]);

    for (int i = 0; i < numSamples; ++i) {
        const auto& curImagePoints = imagePointsList[i];
        const auto& curExtrinsic = extrinsics[i];

        for (int j = 0; j < numPointsEachSample; ++j) {
            cv::Mat objectPointsMat = (cv::Mat_<float>(4, 1) << objectPoints[j].x, objectPoints[j].y, 0., 1.);
            cv::Mat imagePlaneProjection = curExtrinsic * objectPointsMat;

            cv::Mat normalizedProjection = toInHomogenous(imagePlaneProjection.t());
            float r = cv::norm(normalizedProjection);
            cv::Mat pointSensorCoordinate = toInHomogenous((intrinsicMat * imagePlaneProjection).t());

            float du = pointSensorCoordinate.ptr<float>(0)[0] - principalPoint.x;
            float dv = pointSensorCoordinate.ptr<float>(0)[1] - principalPoint.y;

            float r2 = r * r;
            float r4 = r2 * r2;
            D.ptr<float>(i * 2 * numPointsEachSample + 2 * j)[0] = du * r2;
            D.ptr<float>(i * 2 * numPointsEachSample + 2 * j)[1] = du * r4;
            D.ptr<float>(i * 2 * numPointsEachSample + 2 * j + 1)[0] = dv * r2;
            D.ptr<float>(i * 2 * numPointsEachSample + 2 * j + 1)[1] = dv * r4;

            const auto& curImagePoint = curImagePoints[j];
            d.ptr<float>(i * 2 * numPointsEachSample + 2 * j)[0] =
                curImagePoint.x - pointSensorCoordinate.ptr<float>(0)[0];
            d.ptr<float>(i * 2 * numPointsEachSample + 2 * j + 1)[0] =
                curImagePoint.y - pointSensorCoordinate.ptr<float>(0)[1];
        }
    }

    cv::Mat distortionKs;
    cv::solve(D, d, distortionKs, cv::DECOMP_SVD);

    return distortionKs;
}
}  // namespace _cv
