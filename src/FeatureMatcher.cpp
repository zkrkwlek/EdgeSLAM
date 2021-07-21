#include <FeatureMatcher.h>

namespace EdgeSLAM {
	FeatureMatcher::FeatureMatcher(){}
	FeatureMatcher::FeatureMatcher(enum cv::NormTypes type, float fMatchRatio, bool bCrossCheck):norm_type(type), mfMatchRatio(fMatchRatio), mbCrossCheck(bCrossCheck){
	}
	FeatureMatcher::~FeatureMatcher(){}

	int FeatureMatcher::Match(cv::Mat desc1, cv::Mat desc2, std::vector<int>& res1, std::vector<int>& res2, float matchratio) {
		if (matchratio == 0.0)
			matchratio = mfMatchRatio;
		std::vector< std::vector<cv::DMatch> > knn_matches;
		matcher->knnMatch(desc1, desc2, knn_matches, 2);
		GoodMatches(knn_matches, desc1, desc2, res1, res2, matchratio);
		return res1.size();
	}
	void FeatureMatcher::GoodMatches(std::vector< std::vector<cv::DMatch> > matches, cv::Mat desc1, cv::Mat desc2, std::vector<int>& res1, std::vector<int>& res2, float ratio) {
		std::vector<float> matchDists(desc2.rows, FLT_MAX);
		
		std::map<int, int> mapIdx;
		for (size_t i = 0; i < matches.size(); i++)
		{
			if (matches[i][0].distance > ratio * matches[i][1].distance)
				continue;
			int i1 = matches[i][0].queryIdx;
			int i2 = matches[i][0].trainIdx;
			float d1 = matches[i][0].distance;
			float d2 = matchDists[i2];
			if (d2 == FLT_MAX) {
				matchDists[i2] = d1;
				mapIdx[i2] = res1.size();
				res1.push_back(i1);
				res2.push_back(i2);
			}else
			if (d1 < d2) {
				matchDists[i2] = d1;
				int idx = mapIdx[i2];
				res1[idx] = i1;
			}
		}
		
	}

	BFMatcher::BFMatcher(enum cv::NormTypes type, float fMatchRatio, bool bCrossCheck):FeatureMatcher(type, fMatchRatio, bCrossCheck) {
		matcher = cv::makePtr<cv::BFMatcher>(norm_type, bCrossCheck);
	}
	BFMatcher::~BFMatcher(){}

	FlannMatcher::FlannMatcher(enum cv::NormTypes type, float fMatchRatio, bool bCrossCheck):FeatureMatcher(type, fMatchRatio, bCrossCheck) {
		cv::Ptr<cv::flann::IndexParams> indexParams;// = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
		switch (type) {
		case cv::NORM_L2:
			indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(4);
			break;
		case cv::NORM_HAMMING:
		default:
			indexParams = cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1);
			break;
		}
		
		cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(32);//50
		matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
	}
	FlannMatcher::~FlannMatcher(){}
}