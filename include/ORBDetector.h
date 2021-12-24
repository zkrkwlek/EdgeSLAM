#ifndef EDGE_SLAM_ORB_DETECTOR_H
#define EDGE_SLAM_ORB_DETECTOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <FeatureDetector.h>
#include <FeatureInfo.h>
#include <ORBextractor.h>

namespace EdgeSLAM {
	class ORBDetector :public FeatureDetector {
	public:
		ORBDetector(int nFeatures = 1000, float fScaleFactor = 1.2, int nLevels = 8, float fInitThFast = 20, float fMinThFast = 7);
		virtual ~ORBDetector();
		void Compute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);
		void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);
		void init_sigma_level();
	private:
		ORBextractor* detector;
	};
	class ORBInfo : public FeatureInfo {
	public:
		ORBInfo(enum cv::NormTypes type = cv::NORM_L2, float max_dist = 100, float min_dist = 50);
		virtual ~ORBInfo();
	private:
	};

	class ORBDistance :public DescriptorDistance {
	public:
		ORBDistance();
		virtual ~ORBDistance();
		float CalculateDescDistance(cv::Mat a, cv::Mat b);
	private:
	};
}

#endif