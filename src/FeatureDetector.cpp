#include <FeatureDetector.h>
#include <FeatureInfo.h>

namespace EdgeSLAM {
	FeatureDetector::FeatureDetector() {}
	FeatureDetector::FeatureDetector(FeatureInfo* pInfo) :norm_type(pInfo->norm_type), max_descriptor_distance(pInfo->max_descriptor_distance), min_descriptor_distance(pInfo->min_descriptor_distance) {}
	FeatureDetector::~FeatureDetector() {}

	DescriptorDistance::DescriptorDistance(){}
	DescriptorDistance::~DescriptorDistance(){}

	HammingDistance::HammingDistance():DescriptorDistance(){}
	HammingDistance::~HammingDistance(){}
	float HammingDistance::CalculateDescDistance(cv::Mat a, cv::Mat b){
		return 0.0;
	}

	L2Distance::L2Distance() : DescriptorDistance() {}
	L2Distance::~L2Distance() {}
	float L2Distance::CalculateDescDistance(cv::Mat a, cv::Mat b) {
		return 0.0;
	}
}