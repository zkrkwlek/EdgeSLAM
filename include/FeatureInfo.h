#ifndef EDGE_SLAM_FEATURE_INFO_H
#define EDGE_SLAM_FEATURE_INFO_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>

namespace EdgeSLAM {
	class FeatureInfo {
	public:
		//FeatureInfo();
		FeatureInfo(enum cv::NormTypes type, float max_dist, float min_dist);
		virtual ~FeatureInfo();
	public:
		enum cv::NormTypes norm_type;
		float max_descriptor_distance, min_descriptor_distance;
	private:

	};

}

#endif