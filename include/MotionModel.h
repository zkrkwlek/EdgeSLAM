



#ifndef EDGE_SLAM_MOTION_MODEL_H
#define EDGE_SLAM_MOTION_MODEL_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>

namespace EdgeSLAM {
	class CameraPose;
	class MotionModel {
	public:
		MotionModel();
		MotionModel(cv::Mat _T, cv::Mat cov);
		virtual ~MotionModel();
	public:
		cv::Mat deltaT, covariance;
		cv::Mat  predict();
		void update(cv::Mat Tnew);
		void apply_correction();
	private:
		CameraPose *mpCamPose;
	};
}

#endif