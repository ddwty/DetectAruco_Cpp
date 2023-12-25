#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <map>

int main(int argc, char** argv) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        std::cerr << "ERROR opening socket" << std::endl;
        exit(EXIT_FAILURE);
    }

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(12345);
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        std::cerr << "ERROR on inet_pton" << std::endl;
        exit(EXIT_FAILURE);
    }

    if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "ERROR connecting" << std::endl;
        exit(EXIT_FAILURE);
    }

    
    // Camera parameters
    double cx = 655.3664;
    double cy = 367.5246;
    double fx = 971.2252;
    double fy = 970.7470;
    double k1 = 0.0097;
    double k2 = -0.00745;
    double k3 = 0.00;
    double p1 = 0.00;
    double p2 = 0.00;
    // Load camera parameters
    // cv::Mat intrinsic_camera, distortion;

    cv::Mat intrinsic_camera = (cv::Mat_<double>(3, 3) << fx, 0, cx, 
                                                          0, fy, cy, 
                                                          0, 0, 1);

    // Create the distortion coefficients matrix
    cv::Mat distortion = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);


    // Replace with your actual camera parameters file path
    // cv::FileStorage fs("camera_params.yml", cv::FileStorage::READ); 
    // fs["camera_matrix"] >> intrinsic_camera;
    // fs["distortion_coefficients"] >> distortion;
    // fs.release();

   
    cv::Ptr<cv::aruco::Dictionary> arucoDict = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50));
    cv::Ptr<cv::aruco::DetectorParameters> arucoParams = cv::makePtr<cv::aruco::DetectorParameters>();

    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video capture" << std::endl;
        return -1;
    }

    cv::Mat frame, gray;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::Mat kernel = cv::Mat::ones(5, 5, CV_32F) / 25;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::filter2D(gray, gray, -1, kernel);


        std::vector<int> ids;  //用于存储检测到的标记的id
        std::vector<std::vector<cv::Point2f>> corners;  //用于存储检测到的标记的四个角点的坐标
        cv::aruco::detectMarkers(gray, arucoDict, corners, ids, arucoParams);

        std::map<int, cv::Mat> id_to_homogenous_trans_mtx;  //储存每个id对应的齐次变换矩阵
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(frame, corners, ids);
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, 20, intrinsic_camera, distortion, rvecs, tvecs);
            for (int i = 0; i < ids.size(); i++) {

                // std::vector<cv::Vec3d> rvec, tvec;
                // cv::aruco::estimatePoseSingleMarkers(corners[i], 20, intrinsic_camera, distortion, rvec, tvec);

                cv::drawFrameAxes(frame, intrinsic_camera, distortion, rvecs[i], tvecs[i], 10);

                cv::Mat rmat;
                cv::Rodrigues(rvecs[i], rmat);

                cv::Mat homogenous_trans_mtx(4, 4, CV_64F);
                for(int row = 0; row < rmat.rows; row++) {
                    for(int col = 0; col < rmat.cols; col++) {
                        homogenous_trans_mtx.at<double>(row, col) = rmat.at<double>(row, col);
                    }
                    homogenous_trans_mtx.at<double>(row, 3) = tvecs[i][row];
                }
                homogenous_trans_mtx.at<double>(3, 0) = 0;
                homogenous_trans_mtx.at<double>(3, 1) = 0;
                homogenous_trans_mtx.at<double>(3, 2) = 0;
                homogenous_trans_mtx.at<double>(3, 3) = 1;

                id_to_homogenous_trans_mtx[ids[i]] = homogenous_trans_mtx;

                std::cout << "id: " << ids[i] << std::endl;
                std::cout << "homogenous_trans_mtx:\n" << homogenous_trans_mtx << std::endl;

                // Serialize data
                for (const auto& pair : id_to_homogenous_trans_mtx) {
                    int id = pair.first;
                    cv::Mat homogenous_trans_mtx = pair.second;

                    // Convert id and homogenous_trans_mtx to string
                    std::ostringstream oss;
                    oss << "id: " << id << "\n";
                    oss << "homogenous_trans_mtx:\n" << homogenous_trans_mtx << "\n";

                    // Send data
                    std::string serialized_data = oss.str();
                    if (send(sockfd, serialized_data.c_str(), serialized_data.size(), 0) < 0) {
                        std::cerr << "Cannot send data, the server might be down." << std::endl;
                    }
                }
            }   
            // Draw axis for each marker
            // for (int i = 0; i < ids.size(); i++) {
            //     cv::aruco::drawDetectedMarkers(frame, corners, ids);
            // }
        }

        cv::imshow("frame", frame);
        char key = (char) cv::waitKey(1);
        if (key == 27) break; // ESC key
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
