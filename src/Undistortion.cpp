/**
 * @file    Undistortion.cpp
 *
 * @author  btran
 *
 */

#include <cam_intrinsic_calib/Undistortion.hpp>

namespace
{
void getRectangles(const cv::Mat& K, const cv::Mat& distortionParams, const cv::Size& imgSize, cv::Rect_<float>& inner,
                   cv::Rect_<float>& outer)
{
    const int numSamplePoints = 9;
    std::vector<cv::Point2f> distorteds;
    for (int i = 0; i < numSamplePoints; ++i) {
        for (int j = 0; j < numSamplePoints; ++j) {
            distorteds.emplace_back(static_cast<float>(i) * imgSize.width / (numSamplePoints - 1),
                                    static_cast<float>(j) * imgSize.height / (numSamplePoints - 1));
        }
    }
    auto undistorteds = _cv::undistortPoints(distorteds, K, distortionParams, false);

    float outerXMin, outerYMin, outerXMax, outerYMax;
    float innerXMin, innerYMin, innerXMax, innerYMax;
    outerXMin = outerYMin = innerXMax = innerYMax = std::numeric_limits<float>::max();
    outerXMax = outerYMax = innerXMin = innerYMin = std::numeric_limits<float>::lowest();

    for (int i = 0; i < numSamplePoints; ++i) {
        for (int j = 0; j < numSamplePoints; ++j) {
            const auto& undistorted = undistorteds[i * numSamplePoints + j];

            outerXMin = std::min<float>(outerXMin, undistorted.x);
            outerYMin = std::min<float>(outerXMin, undistorted.y);
            outerXMax = std::max<float>(outerXMax, undistorted.x);
            outerYMax = std::max<float>(outerYMax, undistorted.y);

            if (i == 0) {
                innerXMin = std::max<float>(innerXMin, undistorted.x);
            }
            if (i == numSamplePoints - 1) {
                innerXMax = std::min<float>(innerXMax, undistorted.x);
            }
            if (j == 0) {
                innerYMin = std::max<float>(innerYMin, undistorted.y);
            }
            if (j == numSamplePoints - 1) {
                innerYMax = std::min<float>(innerYMax, undistorted.y);
            }
        }
    }
    inner = cv::Rect_<float>(innerXMin, innerYMin, innerXMax - innerXMin, innerYMax - innerYMin);
    outer = cv::Rect_<float>(outerXMin, outerYMin, outerXMax - outerXMin, outerYMax - outerYMin);
}
}  // namespace

namespace _cv
{
float fRadInv(float k0, float k1, float rTilde, int numIterations)
{
    auto f = [](float r, float k0, float k1, float rTilde) {
        return k1 * std::pow(r, 5) + k0 * std::pow(r, 3) + r - rTilde;
    };

    auto fDerivative = [](float r, float k0, float k1) {
        float r2 = r * r;
        return 5 * k1 * r2 * r2 + 3 * k0 * r2 + 1;
    };

    float rInit = rTilde;

    for (int i = 0; i < numIterations; ++i) {
        rInit -= f(rInit, k0, k1, rTilde) / fDerivative(rInit, k0, k1);
    }

    return rInit;
}

cv::Point2f undistortPointNormalizedCoordinate(const cv::Point2f& distorted, const cv::Mat& distortionParams)
{
    float k0 = distortionParams.ptr<float>(0)[0];
    float k1 = distortionParams.ptr<float>(0)[1];
    float rTilde = cv::norm(distorted);
    float r = fRadInv(k0, k1, rTilde);
    float r2 = r * r;
    float Dr = 1 + k0 * r2 + k1 * r2 * r2;

    return cv::Point2f(distorted.x / Dr, distorted.y / Dr);
}

cv::Point2f iterativeUndistortPointNormalizedCoordinate(const cv::Point2f& distorted, const cv::Mat& distortionParams,
                                                        int numIteration)
{
    float k0 = distortionParams.ptr<float>(0)[0];
    float k1 = distortionParams.ptr<float>(0)[1];

    float undistortedXp = distorted.x;
    float undistortedYp = distorted.y;

    for (int i = 0; i < numIteration; ++i) {
        float r2 = undistortedXp * undistortedXp + undistortedYp * undistortedYp;
        float r4 = r2 * r2;
        float Dr = 1 + k0 * r2 + k1 * r4;
        undistortedXp /= Dr;
        undistortedYp /= Dr;
    }

    return cv::Point2f(undistortedXp, undistortedYp);
}

cv::Point2f undistortPoint(const cv::Point2f& sensorCoordDistorted, const cv::Mat& K, const cv::Mat& distortionParams,
                           bool useIterativeMethod)
{
    float K0 = K.ptr<float>(0)[0];
    float K1 = K.ptr<float>(0)[1];
    float K2 = K.ptr<float>(0)[2];
    float K3 = K.ptr<float>(1)[1];
    float K4 = K.ptr<float>(1)[2];

    float yp = (sensorCoordDistorted.y - K4) / K3;
    float xp = (sensorCoordDistorted.x - K2 - K1 * yp) / K0;

    return useIterativeMethod ? iterativeUndistortPointNormalizedCoordinate(cv::Point2f(xp, yp), distortionParams)
                              : undistortPointNormalizedCoordinate(cv::Point2f(xp, yp), distortionParams);
}

std::vector<cv::Point2f> undistortPoints(const std::vector<cv::Point2f>& sensorCoordDistorteds, const cv::Mat& K,
                                         const cv::Mat& distortionParams, bool useIterativeMethod)
{
    std::vector<cv::Point2f> result;
    result.reserve(sensorCoordDistorteds.size());

    for (const auto& distorted : sensorCoordDistorteds) {
        result.emplace_back(undistortPoint(distorted, K, distortionParams, useIterativeMethod));
    }

    return result;
}

cv::Mat getOptimalNewCameraMatrix(const cv::Mat& K, const cv::Mat& distortionParams, const cv::Size& imgSize,
                                  float alpha)
{
    cv::Rect_<float> inner, outer;
    ::getRectangles(K, distortionParams, imgSize, inner, outer);

    float fx0 = (imgSize.width - 1) / inner.width;
    float fy0 = (imgSize.height - 1) / inner.height;
    float cx0 = -fx0 * inner.x;
    float cy0 = -fy0 * inner.y;

    float fx1 = (imgSize.width - 1) / outer.width;
    float fy1 = (imgSize.height - 1) / outer.height;
    float cx1 = -fx1 * outer.x;
    float cy1 = -fy1 * outer.y;

    // clang-format off
    return (cv::Mat_<float>(3, 3) << fx0 * (1 - alpha) + fx1 * alpha, K.ptr<float>(0)[1], cx0 * (1 - alpha) + cx1 * alpha,
                                     0, fy0 * (1 - alpha) + fy1 * alpha, cy0 * (1 - alpha) + cy1 * alpha,
                                     0, 0, 1
           );
    // clang-format on
}

void initUndistortRectifyMap(const cv::Mat& K, const cv::Mat& distortionParams, const cv::Size& imgSize,
                             const cv::Mat& newCameraMatrix, cv::Mat& map1, cv::Mat& map2)
{
    map1.create(imgSize, CV_32FC1);
    map2.create(imgSize, CV_32FC1);

    cv::Mat_<float> A(K);
    cv::Mat_<float> Ar(newCameraMatrix);

    float k0 = reinterpret_cast<float*>(distortionParams.data)[0];
    float k1 = reinterpret_cast<float*>(distortionParams.data)[1];

    for (int i = 0; i < imgSize.height; ++i) {
        float* m1f = reinterpret_cast<float*>(map1.data + map1.step * i);
        float* m2f = reinterpret_cast<float*>(map2.data + map2.step * i);

        for (int j = 0; j < imgSize.width; ++j) {
            float yu = (i - Ar(1, 2)) / Ar(1, 1);
            float xu = (j - Ar(0, 2) - Ar(0, 1) * yu) / Ar(0, 0);
            float r2 = xu * xu + yu * yu;
            float r4 = r2 * r2;
            float Dr = (1 + k0 * r2 + k1 * r4);
            float xd = xu * Dr;
            float yd = yu * Dr;
            float ud = A(0, 0) * xd + A(0, 1) * yd + A(0, 2);
            float vd = A(1, 1) * yd + A(1, 2);
            m1f[j] = ud;
            m2f[j] = vd;
        }
    }
}

void remap(const cv::Mat& src, cv::Mat& dst, const cv::Mat& map1, const cv::Mat& map2)
{
    dst.create(src.size(), src.type());
    uchar* srcData = src.data;
    uchar* dstData = dst.data;

    for (int i = 0; i < src.rows; ++i) {
        float* m1f = reinterpret_cast<float*>(map1.data + map1.step * i);
        float* m2f = reinterpret_cast<float*>(map2.data + map2.step * i);

        for (int j = 0; j < src.cols; ++j) {
            float ud = m1f[j];
            float vd = m2f[j];

            int u0 = ud;
            int v0 = vd;
            int u1 = ud + 1;
            int v1 = vd + 1;
            float du = ud - u0;
            float dv = vd - v0;

            float w00 = (1.0 - du) * (1.0 - dv);
            float w01 = du * (1.0 - dv);
            float w10 = (1.0 - du) * dv;
            float w11 = du * dv;

            for (int c = 0; c < src.channels(); ++c) {
                auto getValue = [srcData, &src, c](int u, int v) {
                    return srcData[int(v) * src.cols * src.channels() + int(u) * src.channels() + c];
                };
                dstData[i * src.cols * src.channels() + j * src.channels() + c] =
                    w00 * getValue(u0, v0) + w01 * getValue(u0, v1) + w10 * getValue(u1, v0) + w11 * getValue(u1, v1);
            }
        }
    }
}
}  // namespace _cv
