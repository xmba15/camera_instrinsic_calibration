/**
 * @file    CalibrationApp.cpp
 *
 * @author  btran
 *
 */

#include <cam_intrinsic_calib/cam_intrinsic_calib.hpp>

#include "AppUtility.hpp"

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: [app] [path/to/json/config]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string CONFIG_PATH = argv[1];
    _cv::CalibrationHandler::Param param = _cv::getCalibrationHandlerParam(CONFIG_PATH);
    _cv::CalibrationHandler calibHandler(param);
    if (calibHandler.allImagePoints().empty()) {
        std::cerr << "failed to get enough image points" << std::endl;
        return EXIT_FAILURE;
    }
    // uncomment the following line to draw detected chessboard corners in images written under /tmp
    // calibHandler.drawChessboardCorners();

    cv::Mat K, distortionParams;
    std::vector<cv::Mat> extrinsics;

    calibHandler.run(K, distortionParams, extrinsics);
    calibHandler.drawReporjections(K, distortionParams, extrinsics);

    cv::Affine3f camPose;
    double scale = 0.1;
    auto cameraWindow = ::initializeWindowWithCamera(camPose, cv::Matx33d(K), scale);

    auto colors = _cv::utils::generateColorCharts(extrinsics.size());

    for (std::size_t i = 0; i < extrinsics.size(); ++i) {
        cv::Affine3f boardPose(extrinsics[i](cv::Rect(0, 0, 3, 3)), extrinsics[i].col(3));
        cv::Affine3f boardPoseInWorld = camPose * boardPose;
        cv::Mat rotation(boardPoseInWorld.rotation());
        cv::Vec3d translation(boardPoseInWorld.translation());
        cv::Vec3d xAxis(rotation.col(0));
        cv::Vec3d yAxis(rotation.col(1));
        cv::Vec3d zAxis(rotation.col(2));

        // clang-format off
        cv::Point3d center = translation +
                             xAxis * calibHandler.param().numCol * calibHandler.param().squareSize / 2 +
                             yAxis * calibHandler.param().numRow * calibHandler.param().squareSize / 2;
        // clang-format on

        cameraWindow.showWidget("board" + std::to_string(i),
                                cv::viz::WPlane(cv::Point3d(center), zAxis, yAxis, cv::Size2d(0.2, 0.2), colors[i]));
        cameraWindow.showWidget("text" + std::to_string(i),
                                cv::viz::WText3D(std::to_string(i), cv::Point3d(translation), 0.01, false, colors[i]));
    }

    cameraWindow.spin();

    return EXIT_SUCCESS;
}
