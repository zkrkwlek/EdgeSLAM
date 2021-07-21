#ifndef EDGE_SLAM_CONVERTER_H
#define EDGE_SLAM_CONVERTER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include<Eigen/Dense>
#include"g2o/types/types_six_dof_expmap.h"
#include"g2o/types/types_seven_dof_expmap.h"

namespace EdgeSLAM {
	class Converter
	{
	public:
		static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);

		static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
		static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);

		static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
		static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
		static cv::Mat toCvMat(const Eigen::Matrix<double, 4, 4> &m);
		static cv::Mat toCvMat(const Eigen::Matrix3d &m);
		static cv::Mat toCvMat(const Eigen::Matrix<double, 3, 1> &m);
		static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);

		static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Mat &cvVector);
		static Eigen::Matrix<double, 3, 1> toVector3d(const cv::Point3f &cvPoint);
		static Eigen::Matrix<double, 3, 3> toMatrix3d(const cv::Mat &cvMat3);

		static std::vector<float> toQuaternion(const cv::Mat &M);
	};
}

#endif