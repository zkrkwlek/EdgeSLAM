#ifndef EDGE_SLAM_SEARCH_POINTS_H
#define EDGE_SLAM_SEARCH_POINTS_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
//#include "DBoW2/BowVector.h"
//#include "DBoW2/FeatureVector.h"
#include <DBoW3.h>

namespace EdgeSLAM {
	class Frame;
	class KeyFrame;
	class MapPoint;
	class TrackPoint;
	class FeatureTracker;
	class SearchPoints {
	public:
		static const int HISTO_LENGTH;
		static int SearchBySim3(FeatureTracker* pFeatureTracker, KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint*> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, float thMaxDesc, float thRadius = 7.5);
		static int SearchKeyFrameByBoW(FeatureTracker* pFeatureTracker, KeyFrame* pKF1, KeyFrame *pKF2, std::vector<MapPoint*> &vpMapPointMatches, float thMinDesc, float thMatchRatio, bool bCheckOri = true);
		static int SearchFrameByBoW(FeatureTracker* pFeatureTracker, KeyFrame* pKF, Frame *F, std::vector<MapPoint*> &vpMapPointMatches, float thMinDesc, float thMatchRatio, bool bCheckOri = true);
		static int SearchFrameByProjection(Frame* prev, Frame* curr, float thMaxDesc, float thMinDesc, float thProjection = 15, bool bCheckOri = true);
		static int SearchFrameByProjection(FeatureTracker* pFeatureTracker,Frame *pF, KeyFrame *pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist, bool bCheckOri = true);
		static int SearchKeyByProjection(FeatureTracker* pFeatureTracker, KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, float thMinDesc, float thRadius = 10.0);
		static int SearchMapByProjection(Frame *F, const std::vector<MapPoint*> &vpMapPoints, const std::vector<TrackPoint*> &vpTrackPoints, float thMaxDesc, float thMinDesc, float thRadius = 1.0, float thMatchRatio = 0.8, bool bCheckOri = true);
		static int SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12, std::vector<std::pair<size_t, size_t> > &vMatchedPairs, float thMaxDesc, float thMinDesc, float thRatio = 0.6, bool bCheckOri = false);
		static int Fuse(KeyFrame *pKF, const std::vector<MapPoint *> &vpMapPoints, float thMaxDesc, float thMinDesc, const float th = 3.0);
		static int Fuse(FeatureTracker* pFeatureTracker, KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpPoints, std::vector<MapPoint *> &vpReplacePoint, float thMinDesc, float thRadius);
		static bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame* pKF2);
		static void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
		static float RadiusByViewingCos(const float &viewCos);
	};
}
#endif

//7, trackergw