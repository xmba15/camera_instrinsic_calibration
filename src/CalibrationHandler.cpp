/**
 * @file    CalibrationHandler.cpp
 *
 * @author  btran
 *
 */

#include <algorithm>

#include <cam_intrinsic_calib/CalibrationHandler.hpp>
#include <cam_intrinsic_calib/utils/BasicUtils.hpp>
#include <cam_intrinsic_calib/utils/CalibUtils.hpp>

namespace _cv
{
CalibrationHandler::CalibrationHandler(const Param& param)
    : m_param(param)
    , m_patternSize(m_param.numCol, m_param.numRow)
{
    validate<Param>(param);
    m_imageList = parseMetaDataFile(param.imageListFile);
    m_objectPoints = generateObjectPoints(m_param.numRow, m_param.numCol, m_param.squareSize);
    m_allImagePoints = this->getImagePoints();
}

void CalibrationHandler::run(cv::Mat& K, cv::Mat& distortionParams, std::vector<cv::Mat>& extrinsics) const
{
    std::vector<cv::Mat> homographies = calculateHomographies(m_objectPoints, m_allImagePoints);
    K = getCameraIntrinsicFromHomographies(homographies);
    extrinsics = getCameraExtrinsics(homographies, K);
    distortionParams = estimateRadialLensDistortion(m_objectPoints, m_allImagePoints, K, extrinsics);
    refineAll(m_objectPoints, m_allImagePoints, K, distortionParams, extrinsics);
}

void CalibrationHandler::drawChessboardCorners() const
{
    std::string outputPath = "/tmp";
    for (std::size_t i = 0; i < m_imageList.size(); ++i) {
        cv::Mat image = cv::imread(m_param.imagePath + "/" + m_imageList[i]);
        cv::drawChessboardCorners(image, m_patternSize, cv::Mat(m_allImagePoints[i]), true);
        cv::imwrite(outputPath + "/" + m_imageList[i], image);
    }
}

void CalibrationHandler::drawReporjections(const cv::Mat& K, const cv::Mat& distortionParams,
                                           const std::vector<cv::Mat>& extrinsics) const
{
    assert(extrinsics.size() == m_imageList.size());

    std::string outputPath = "/tmp";
    for (std::size_t i = 0; i < m_imageList.size(); ++i) {
        cv::Mat image = cv::imread(m_param.imagePath + "/" + m_imageList[i]);
        this->drawReprojectionUtil(image, K, distortionParams, extrinsics[i], m_objectPoints);
        cv::Point3f origin(m_objectPoints.front().x, m_objectPoints.front().y, 0);
        this->draw3DCoordinate(image, origin, K, distortionParams, extrinsics[i]);
        cv::imwrite(outputPath + "/reprojected_" + m_imageList[i], image);
    }
}

void CalibrationHandler::drawReprojectionUtil(cv::Mat& image, const cv::Mat& K, const cv::Mat& distortionParams,
                                              const cv::Mat& extrinsic,
                                              const std::vector<cv::Point2f>& m_objectPoints) const
{
    int numPoints = m_objectPoints.size();
    for (int i = 0; i < numPoints; ++i) {
        cv::Point3f curObjectPoint(m_objectPoints[i].x, m_objectPoints[i].y, 0);
        cv::Point2f sensorPoint = this->reproject(curObjectPoint, K, distortionParams, extrinsic);
        cv::circle(image, sensorPoint, 3, cv::Scalar(62, 22, 156), -1);
    }
}

void CalibrationHandler::draw3DCoordinate(cv::Mat& image, const cv::Point3f& origin, const cv::Mat& K,
                                          const cv::Mat& distortionParams, const cv::Mat& extrinsic, float length) const
{
    cv::Point2f projectedOrigin = this->reproject(origin, K, distortionParams, extrinsic);
    cv::Point3f xAxis = origin + length * cv::Point3f(1, 0, 0);
    cv::Point3f yAxis = origin + length * cv::Point3f(0, 1, 0);
    cv::Point3f zAxis = origin + length * cv::Point3f(0, 0, 1);

    cv::Point2f projectedxAxis = this->reproject(xAxis, K, distortionParams, extrinsic);
    cv::Point2f projectedyAxis = this->reproject(yAxis, K, distortionParams, extrinsic);
    cv::Point2f projectedzAxis = this->reproject(zAxis, K, distortionParams, extrinsic);
    cv::line(image, projectedOrigin, projectedxAxis, cv::Scalar(0, 0, 255), 3);
    cv::line(image, projectedOrigin, projectedyAxis, cv::Scalar(0, 255, 0), 3);
    cv::line(image, projectedOrigin, projectedzAxis, cv::Scalar(255, 0, 0), 3);
}

cv::Point2f CalibrationHandler::reproject(const cv::Point3f& objectPoint, const cv::Mat& K,
                                          const cv::Mat& distortionParams, const cv::Mat& extrinsic) const
{
    cv::Mat curObjectPointMat = (cv::Mat_<float>(4, 1) << objectPoint.x, objectPoint.y, objectPoint.z, 1);
    cv::Mat normalizedPoint = toInHomogenous((extrinsic * curObjectPointMat).t());
    float& xp = normalizedPoint.ptr<float>(0)[0];
    float& yp = normalizedPoint.ptr<float>(0)[1];
    float r2 = xp * xp + yp * yp;
    float r4 = r2 * r2;
    float k0 = distortionParams.ptr<float>(0)[0];
    float k1 = distortionParams.ptr<float>(0)[1];
    float Dr = 1 + k0 * r2 + k1 * r4;
    xp *= Dr;
    yp *= Dr;
    cv::Mat sensorPointMat = (K(cv::Rect(0, 0, 3, 2)) * toHomogenous(normalizedPoint).t()).t();
    return cv::Point2f(sensorPointMat.ptr<float>(0)[0], sensorPointMat.ptr<float>(0)[1]);
}

std::vector<std::vector<cv::Point2f>> CalibrationHandler::getImagePoints()
{
    std::vector<std::vector<cv::Point2f>> imagePoints;
    imagePoints.reserve(m_imageList.size());

    const cv::Size subPixWinSize(5, 5);
    const cv::TermCriteria termimateCrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    for (auto it = m_imageList.begin(); it != m_imageList.end();) {
        cv::Mat gray = cv::imread(m_param.imagePath + "/" + *it, 0);

        if (m_imgSize.width * m_imgSize.height == 0) {
            m_imgSize = gray.size();
        }

        if (gray.empty()) {
            throw std::runtime_error("failed to read " + *it);
        }
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(gray, m_patternSize, corners);
        if (!found) {
            std::cerr << "failed to find corners on image " + *it << std::endl;
            it = m_imageList.erase(it);
            continue;
        }

        cv::cornerSubPix(gray, corners, subPixWinSize, cv::Size(-1, -1), termimateCrit);
        imagePoints.emplace_back(corners);
        ++it;
    }

    return imagePoints;
}

CalibrationHandler::Param getCalibrationHandlerParam(const std::string& jsonPath)
{
    rapidjson::Document jsonDoc = readFromJsonFile(jsonPath);
    CalibrationHandler::Param param;
    param.imagePath = getValueAs<std::string>(jsonDoc, "image_path");
    param.imageListFile = getValueAs<std::string>(jsonDoc, "image_list_file");
    param.numRow = getValueAs<int>(jsonDoc, "num_row");
    param.numCol = getValueAs<int>(jsonDoc, "num_col");
    param.squareSize = getValueAs<float>(jsonDoc, "square_size");

    return param;
}

template <> void validate<CalibrationHandler::Param>(const CalibrationHandler::Param& param)
{
    if (param.imagePath.empty()) {
        throw std::runtime_error("empty path to images");
    }

    if (param.imageListFile.empty()) {
        throw std::runtime_error("empty file that stores image list");
    }

    if (param.numRow <= 0) {
        throw std::runtime_error("invalid number of rows");
    }

    if (param.numCol <= 0) {
        throw std::runtime_error("invalid number of columns");
    }

    if (param.squareSize <= 0) {
        throw std::runtime_error("invalid square size");
    }
}
}  // namespace _cv
