#ifndef EDGE_SLAM_FEATURE_TRACKER_H
#define EDGE_SLAM_FEATURE_TRACKER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>

namespace EdgeSLAM {
	class FeatureDetector;
	class FeatureMatcher;
	class FeatureTracker {
	public:
		FeatureTracker();
		FeatureTracker(FeatureDetector* pD, FeatureMatcher* pM, int nFeatures, int nLevel, float fSacleFactor, float fMatchRatio = 0.7);
		virtual ~FeatureTracker();
	public:
		/*template<typename T>
		static T detector;*/
		virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);
		virtual int match(cv::Mat desc1, cv::Mat desc2, std::vector<int>& res1, std::vector<int>& res2, float matchratio = 0.0);
		virtual float DescriptorDistance(cv::Mat a, cv::Mat b);
		virtual void track();
	public:
		float max_descriptor_distance, min_descriptor_distance;
		int mnFeatures;
		int mnLevels;
		float mfScaleFactor;
		float mfMatchRatio;
		FeatureDetector* detector;
		FeatureMatcher* matcher;
	};

	class FlannFeatureTracker : public FeatureTracker {
	public:
		FlannFeatureTracker(int nFeatures = 1000, int nLevel = 8, float fSacleFactor = 1.2, float fMatchRatio = 0.7);
		virtual ~FlannFeatureTracker();
	private:
	};
}

#endif