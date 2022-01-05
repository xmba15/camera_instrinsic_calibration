/**
 * @file    UndistortionApp.cpp
 *
 * @author  btran
 *
 */

#include <cam_intrinsic_calib/cam_intrinsic_calib.hpp>

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: [app] [path/to/sample/image]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string IMAGE_PATH = argv[1];
    cv::Mat sampleImage = cv::imread(IMAGE_PATH);
    if (sampleImage.empty()) {
        std::cerr << "failed to load: " << IMAGE_PATH << std::endl;
        return EXIT_FAILURE;
    }

    // camera matrix and distortion params obtained from calibration app
    // clang-format off
    cv::Mat K = (cv::Mat_<float>(3, 3) << 461.69708, -0.30813545, 316.49506,
                                          0, 461.97879, 189.62727,
                                          0, 0, 1);
    // clang-format on
    cv::Mat distortionParams = (cv::Mat_<float>(2, 1) << 0.11387298, -0.20481244);
    const cv::Size& imgSize = sampleImage.size();

    float alpha = 0;
    cv::Mat newCameraMatrix = _cv::getOptimalNewCameraMatrix(K, distortionParams, imgSize, alpha);
    cv::Mat map1, map2;
    _cv::initUndistortRectifyMap(K, distortionParams, imgSize, newCameraMatrix, map1, map2);
    cv::Mat undistorted;

    // _cv::remap(sampleImage, undistorted, map1, map2);
    cv::remap(sampleImage, undistorted, map1, map2, cv::INTER_AREA);  // use opencv's remap for now
    cv::hconcat(sampleImage, undistorted, undistorted);
    cv::imshow("undistored image", undistorted);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
