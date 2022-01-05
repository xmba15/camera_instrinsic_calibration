/**
 * @file    CalibrationHandler.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Extrinsics.hpp"
#include "Homography.hpp"
#include "LensDistortionModel.hpp"
#include "TotalRefinement.hpp"
#include "utils/utils.hpp"

namespace _cv
{
class CalibrationHandler
{
 public:
    struct Param {
        std::string imagePath;      // path to the directory that stores images
        std::string imageListFile;  // file that stores image list
        int numRow;                 // num (inner) row of the checkerboard
        int numCol;                 // num (inner) column of the checkerboard
        float squareSize;
    };

 public:
    explicit CalibrationHandler(const Param& param);

    void run(cv::Mat& K, cv::Mat& distortionParams, std::vector<cv::Mat>& extrinsics) const;

    /**
     *   \file debug function to draw detected chess board corners
     */
    void drawChessboardCorners() const;

    void drawReporjections(const cv::Mat& K, const cv::Mat& distortionParams,
                           const std::vector<cv::Mat>& extrinsics) const;

    const auto& allImagePoints() const
    {
        return m_allImagePoints;
    }

    const auto& param() const
    {
        return m_param;
    }

    const auto& imgSize() const
    {
        return m_imgSize;
    }

 private:
    void drawReprojectionUtil(cv::Mat& image, const cv::Mat& K, const cv::Mat& distortionParams,
                              const cv::Mat& extrinsic, const std::vector<cv::Point2f>& m_objectPoints) const;

    void draw3DCoordinate(cv::Mat& image, const cv::Point3f& origin, const cv::Mat& K, const cv::Mat& distortionParams,
                          const cv::Mat& extrinsic, float length = 0.05) const;

    cv::Point2f reproject(const cv::Point3f& objectPoint, const cv::Mat& K, const cv::Mat& distortionParams,
                          const cv::Mat& extrinsic) const;

    /**
     *   \file get corner points on checker board images
     *
     */
    std::vector<std::vector<cv::Point2f>> getImagePoints();

 private:
    Param m_param;
    cv::Size m_patternSize;
    cv::Size m_imgSize;

    std::vector<std::string> m_imageList;
    std::vector<cv::Point2f> m_objectPoints;  // object points for one image
    std::vector<std::vector<cv::Point2f>> m_allImagePoints;
};

CalibrationHandler::Param getCalibrationHandlerParam(const std::string& jsonPath);
template <> void validate<CalibrationHandler::Param>(const CalibrationHandler::Param& param);

}  // namespace _cv
