#ifndef EDGE_SLAM_CAMERA_POSE_H
#define EDGE_SLAM_CAMERA_POSE_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace EdgeSLAM {
	class CameraPose {
	public:
		CameraPose();
		CameraPose(cv::Mat T);
		virtual ~CameraPose();
	public:
		void SetPose(cv::Mat T);
		cv::Mat GetPose();
		void GetPose(cv::Mat& R, cv::Mat& t);
		cv::Mat GetCenter();
		cv::Mat GetInversePose();
		cv::Mat GetRotation();
		cv::Mat GetTranslation();

	private:
		//frorm world to camera
		std::mutex mMutexPose;
		cv::Mat Tcw; //4x4
		cv::Mat Rcw; //3x3
		cv::Mat tcw; //3x1
		cv::Mat Ow;  //3x1
	};
}

#endif