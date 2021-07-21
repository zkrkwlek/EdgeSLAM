#include <FeatureInfo.h>

namespace EdgeSLAM {
	//FeatureInfo::FeatureInfo(){}
	FeatureInfo::FeatureInfo(enum cv::NormTypes type, float max_dist, float min_dist):
		norm_type(type), max_descriptor_distance(max_dist), min_descriptor_distance(min_dist) {}
	FeatureInfo::~FeatureInfo(){}
}