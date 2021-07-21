#ifndef EDGE_SLAM_MATCHER_H
#define EDGE_SLAM_MATCHER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace EdgeSLAM {
	
	class FeatureMatcher {
	public:
		FeatureMatcher();
		FeatureMatcher(enum cv::NormTypes type, float fMatchRatio, bool bCrossCheck = false);
		virtual ~FeatureMatcher();
	public:
		int Match(cv::Mat desc1, cv::Mat desc2, std::vector<int>& res1, std::vector<int>& res2, float matchratio = 0.0);
	private:
		void GoodMatches(std::vector< std::vector<cv::DMatch> > matches, cv::Mat desc1, cv::Mat desc2, std::vector<int>& res1, std::vector<int>& res2, float ratio);
	public:
		cv::Ptr<cv::DescriptorMatcher> matcher;
		enum cv::NormTypes norm_type;
		bool mbCrossCheck;
		float mfMatchRatio;
	private:

	};

	class BFMatcher : public FeatureMatcher {
	public:
		BFMatcher(enum cv::NormTypes type, float fMatchRatio, bool bCrossCheck = false);
		virtual ~BFMatcher();
	};

	class FlannMatcher :public FeatureMatcher {
	public:
		FlannMatcher(enum cv::NormTypes type, float fMatchRatio, bool bCrossCheck = false);
		virtual ~FlannMatcher();
	};

}

#endif