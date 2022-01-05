/**
 * @file    TotalRefinement.cpp
 *
 * @author  btran
 *
 */

#include <cam_intrinsic_calib/TotalRefinement.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace
{
struct TotalRefinementResidual {
    TotalRefinementResidual(const cv::Point2f& objectPoint, const cv::Point2f& imagePoint)
        : m_objectPoint(objectPoint)
        , m_imagePoint(imagePoint)
    {
    }

    template <typename T>
    bool operator()(const T* const K, const T* const distortionParams, const T* const rotationVec,
                    const T* const translation, T* residuals) const
    {
        T objectPoint[3] = {static_cast<T>(m_objectPoint.x), static_cast<T>(m_objectPoint.y), static_cast<T>(0.)};
        T p[3];
        ceres::AngleAxisRotatePoint(rotationVec, objectPoint, p);
        for (int i = 0; i < 3; ++i) {
            p[i] += translation[i];
        }

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        T r2 = xp * xp + yp * yp;
        T r4 = r2 * r2;

        const T& k0 = distortionParams[0];
        const T& k1 = distortionParams[1];

        T Dr = (static_cast<T>(1) + k0 * r2 + k1 * r4);
        xp = xp * Dr;
        yp = yp * Dr;
        T u = K[0] * xp + K[1] * yp + K[2];
        T v = K[3] * yp + K[4];

        residuals[0] = static_cast<T>(m_imagePoint.x) - u;
        residuals[1] = static_cast<T>(m_imagePoint.y) - v;

        return true;
    }

 private:
    template <typename T> void getInverseAngleAxis(const T* const rotationVec, T* invRotationVec) const
    {
        T rotationMatData[9];
        ceres::AngleAxisToRotationMatrix(rotationVec, rotationMatData);
        Eigen::Matrix<T, 3, 3> rotationMat(rotationMatData);
        rotationMat.transposeInPlace();
        ceres::RotationMatrixToAngleAxis(rotationMat.data(), invRotationVec);
    }

 private:
    const cv::Point2f& m_objectPoint;
    const cv::Point2f& m_imagePoint;
};
}  // namespace

namespace _cv
{
void refineAll(const std::vector<cv::Point2f>& objectPoints,
               const std::vector<std::vector<cv::Point2f>>& imagePointsList, cv::Mat& intrinsicMat,
               cv::Mat& distortionParams, std::vector<cv::Mat>& extrinsics)
{
    int numSamples = extrinsics.size();
    int numPointsEachSample = objectPoints.size();

    std::vector<double> kVec = {intrinsicMat.ptr<float>(0)[0], intrinsicMat.ptr<float>(0)[1],
                                intrinsicMat.ptr<float>(0)[2], intrinsicMat.ptr<float>(1)[1],
                                intrinsicMat.ptr<float>(1)[2]};
    std::vector<double> dVec = {distortionParams.ptr<float>(0)[0], distortionParams.ptr<float>(0)[1]};

    std::vector<cv::Mat> rotationVecs(numSamples, cv::Mat::zeros(3, 1, CV_64FC1));
    std::vector<cv::Mat> translationVecs(numSamples, cv::Mat::zeros(3, 1, CV_64FC1));

    ceres::Problem prob;
    for (int i = 0; i < numSamples; ++i) {
        const auto& curExtrinsic = extrinsics[i];
        const auto& rotationMat = curExtrinsic(cv::Rect(0, 0, 3, 3));
        cv::Rodrigues(rotationMat, rotationVecs[i]);
        curExtrinsic.col(3).copyTo(translationVecs[i]);

        rotationVecs[i].convertTo(rotationVecs[i], CV_64FC1);
        curExtrinsic.col(3).convertTo(translationVecs[i], CV_64FC1);

        const auto& curImagePoints = imagePointsList[i];

        for (int j = 0; j < numPointsEachSample; ++j) {
            prob.AddResidualBlock(new ceres::AutoDiffCostFunction<TotalRefinementResidual, 2, 5, 2, 3, 3>(
                                      new TotalRefinementResidual(objectPoints[j], curImagePoints[j])),
                                  nullptr, kVec.data(), dVec.data(), rotationVecs[i].ptr<double>(),
                                  translationVecs[i].ptr<double>());
        }
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_SCHUR;

#ifdef DEBUG
    options.minimizer_progress_to_stdout = true;
#endif

    ceres::Solver::Summary summary;
    ceres::Solve(options, &prob, &summary);

#ifdef DEBUG
    std::cout << summary.BriefReport() << std::endl;
#endif

    // copy optimized data
    std::copy(dVec.begin(), dVec.end(), distortionParams.ptr<float>());
    intrinsicMat = (cv::Mat_<float>(3, 3) << kVec[0], kVec[1], kVec[2], 0, kVec[3], kVec[4], 0, 0, 1);

    for (int i = 0; i < numSamples; ++i) {
        auto& curExtrinsic = extrinsics[i];
        cv::Mat rotationMat;
        cv::Rodrigues(rotationVecs[i], rotationMat);
        rotationMat.convertTo(curExtrinsic(cv::Rect(0, 0, 3, 3)), curExtrinsic.type());
        translationVecs[i].convertTo(curExtrinsic.col(3), curExtrinsic.type());
    }
}
}  // namespace _cv
