#include <FeatureTracker.h>
#include <FeatureDetector.h>
#include <FeatureInfo.h>
#include <FeatureMatcher.h>
#include <ORBDetector.h>
namespace EdgeSLAM {
	FeatureTracker::FeatureTracker(){}
	FeatureTracker::FeatureTracker(FeatureDetector* pD, FeatureMatcher* pM, int nFeatures, int nLevel, float fScaleFactor, float fMatchRatio):
		detector(pD), matcher(pM),mnFeatures(nFeatures),mnLevels(nLevel), mfScaleFactor(fScaleFactor), mfMatchRatio(fMatchRatio), max_descriptor_distance(pD->max_descriptor_distance), min_descriptor_distance(pD->min_descriptor_distance)
		
	{
	}
	FeatureTracker::~FeatureTracker(){}

	void FeatureTracker::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) {
		detector->detectAndCompute(image, mask, keypoints, descriptors);
	}
	int FeatureTracker::match(cv::Mat desc1, cv::Mat desc2, std::vector<int>& res1, std::vector<int>& res2, float matchratio) {
		if (matchratio == 0.0)
			matchratio = mfMatchRatio;
		return matcher->Match(desc1, desc2, res1, res2, matchratio);
	}
	float FeatureTracker::DescriptorDistance(cv::Mat a, cv::Mat b) {
		return detector->mpDistance->CalculateDescDistance(a, b);
	}

	void FeatureTracker::track() {
	}

	FlannFeatureTracker::FlannFeatureTracker(int nFeatures, int nLevel, float fSacleFactor, float fMatchRatio):FeatureTracker(new ORBDetector(nFeatures, fSacleFactor, nLevel), new FlannMatcher(cv::NORM_HAMMING, fMatchRatio), nFeatures, nLevel, fSacleFactor, fMatchRatio) {
	}
	FlannFeatureTracker::~FlannFeatureTracker(){}
}