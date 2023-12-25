#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem> // C++17 standard, use <experimental/filesystem> for older compilers
#include <chrono>
#include <thread>
#include <unistd.h>
#include <mach-o/dyld.h>  // For MacOS

namespace fs = std::__fs::filesystem; // Adjust if using <experimental/filesystem>

void capture_calibration_images(int camera_id, int num_images, int delay, std::vector<cv::Mat>& images);
void calibrate_camera(const std::vector<cv::Mat>& images);


int main() {
    int camera_id = 0;
    int num_images = 0;
    int delay = 1; // Delay in seconds between capturing each image

    // 
    char path[1024];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) {
        std::cout << "Executable path is: " << path << std::endl;
    } else {
        std::cerr << "Buffer too small; need size " << size << std::endl;
        return 1;
    }

    std::string pathStr = path;
    std::string directoryPath = pathStr.substr(0, pathStr.find_last_of("/")); // 提取目录路径

    // 改变当前工作目录
    if (chdir(directoryPath.c_str()) != 0) {
        std::cerr << "Failed to change directory to " << directoryPath << std::endl;
        return 1;
    }

    std::string folder = "cali_imgs";
    if (std::__fs::filesystem::exists(folder)) {
        // 删除文件夹中的所有文件
        for (const auto & entry : std::__fs::filesystem::directory_iterator(folder)) {
            try {
                std::__fs::filesystem::remove_all(entry.path());
            } catch (std::__fs::filesystem::filesystem_error& e) {
                std::cerr << "Failed to delete " << entry.path() << ". Reason: " << e.what() << std::endl;
            }
        }
    } else {
        std::__fs::filesystem::create_directories(folder);
    }
    
    std::cout << "Enter the number of images you want to capture: ";
    std::cin >> num_images;

    std::vector<cv::Mat> images;
    capture_calibration_images(camera_id, num_images, delay, images);
    calibrate_camera(images);

    return 0;
}



void capture_calibration_images(int camera_id, int num_images, int delay, std::vector<cv::Mat>& images) {
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open the camera" << std::endl;
        return;
    }

    int photo_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    while (cap.isOpened() && photo_count < num_images) {
        cv::Mat frame, overlay;
        bool ret = cap.read(frame);
        if (!ret) break;

        // Calculate the remaining time and the end angle for the ellipse
       // Inside the capture loop
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
        double remaining_time = std::max(0.0, static_cast<double>(delay * 1000) - elapsed_time);
        double end_angle = 360.0 * remaining_time / static_cast<double>(delay * 1000);

        // Create an overlay with a white ellipse
        overlay = frame.clone();
        cv::ellipse(overlay, cv::Point(frame.cols / 2, frame.rows / 2), 
                    cv::Size(frame.rows / 2, frame.rows / 2), -90, 0, end_angle, 
                    cv::Scalar(255, 255, 255), -1);

        // Blend the overlay with the original frame
        double alpha = 0.5;
        cv::addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame);

        // Add text to the frame
        std::string text = "Photo " + std::to_string(photo_count + 1) + "/" + std::to_string(num_images);
        cv::putText(frame, text, cv::Point(frame.cols - 200, frame.rows - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Calibration", frame);

         // ESC to quit
        if ((cv::waitKey(1) & 0xFF) == 27) break;

        if (remaining_time == 0) {
            images.push_back(frame.clone());
            std::cout << "Image captured: " << photo_count << std::endl;
            cv::imwrite("cali_imgs/calibration_" + std::to_string(photo_count) + ".png", frame);
            photo_count++;
            start_time = std::chrono::steady_clock::now();

            // Display a white frame for a brief moment
            cv::Mat white_frame(frame.size(), frame.type(), cv::Scalar(255, 255, 255));
            cv::imshow("Calibration", white_frame);
            cv::waitKey(100); // Display for 100 milliseconds
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

void calibrate_camera(const std::vector<cv::Mat>& images) {
    int num_horizontal = 11;
    int num_vertical = 8;
    double grid_width = 0.020; // meter

    std::vector<std::vector<cv::Point3f>> objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints;
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < num_horizontal * num_vertical; ++i) {
        objp.emplace_back((i / num_horizontal) * grid_width, (i % num_horizontal) * grid_width, 0);
    }

    for (const auto& img : images) {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners; 
        bool found = cv::findChessboardCorners(gray, cv::Size(num_horizontal, num_vertical), corners);

        if (found) {
            cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
            imgpoints.push_back(corners);
            objpoints.push_back(objp);
            cv::drawChessboardCorners(img, cv::Size(num_horizontal, num_vertical), corners, found);
            cv::imshow("Chessboard Corners", img);
            cv::waitKey(500);
        }
    }

    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objpoints, imgpoints, cv::Size(1280, 720), // Use your camera resolution
                                     cameraMatrix, distCoeffs, rvecs, tvecs);

    std::cout << "Reprojection error: " << rms << std::endl;
    if (rms < 1.0) {
        std::cout << "Camera calibrated successfully" << std::endl;
        // Save the camera parameters
        cv::FileStorage fs("camera_params.yml", cv::FileStorage::WRITE);
        fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
        fs.release();
    } else {
        std::cout << "Calibration failed" << std::endl;
    }
}
