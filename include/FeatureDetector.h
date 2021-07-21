#ifndef EDGE_SLAM_FEATURE_DETECTOR_H
#define EDGE_SLAM_FEATURE_DETECTOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace EdgeSLAM {
	class FeatureInfo;
	class DescriptorDistance;
	class FeatureDetector {
	public:
		FeatureDetector();
		FeatureDetector(FeatureInfo* pInfo);
		virtual ~FeatureDetector();
	public:
		virtual void init_sigma_level() {}
		virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors){}
	public:
		enum cv::NormTypes norm_type;
		float max_descriptor_distance;
		float min_descriptor_distance;
		DescriptorDistance* mpDistance;
	public:
		int mnScaleLevels;
		float mfScaleFactor;
		float mfLogScaleFactor;
		std::vector<float> mvScaleFactors;
		std::vector<float> mvInvScaleFactors;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;
	private:

	};

	class DescriptorDistance {
	public:
		DescriptorDistance();
		virtual ~DescriptorDistance();
		virtual float CalculateDescDistance(cv::Mat a, cv::Mat b) { return 0.0; }
	private:
	};
	class HammingDistance :public DescriptorDistance {
	public:
		HammingDistance();
		virtual ~HammingDistance();
		float CalculateDescDistance(cv::Mat a, cv::Mat b);
	private:
	};

	class L2Distance :public DescriptorDistance {
	public:
		L2Distance();
		virtual ~L2Distance();
		float CalculateDescDistance(cv::Mat a, cv::Mat b);
	private:
	};
}

#endif

/*

self.inv_scale_factor = 1./self.scale_factor
self.log_scale_factor = math.log(self.scale_factor)

self.scale_factors = np.zeros(num_levels)
self.inv_scale_factors = np.zeros(num_levels)

self.level_sigmas2 = np.zeros(num_levels)
self.level_sigmas = np.zeros(num_levels)
self.inv_level_sigmas2 = np.zeros(num_levels)




*/