#ifndef EDGE_SLAM_UTILS_H
#define EDGE_SLAM_UTILS_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace EdgeSLAM {
	class Utils {
	public:
		static cv::Mat SkewSymmetricMatrix(const cv::Mat &v);
		static cv::Mat ComputeF12(cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat K1, cv::Mat K2);
		static cv::Mat RotationMatrixFromEulerAngles(float a, float b, float c, std::string str);
	private:
		static cv::Mat RotationMatrixFromEulerAngle(float a, char c);
	};
}
#endif