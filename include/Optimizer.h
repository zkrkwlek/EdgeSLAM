#ifndef EDGE_SLAM_OPTIMIZER_H
#define EDGE_SLAM_OPTIMIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>

namespace EdgeSLAM {
	class MapPoint;
	class Frame;
	class KeyFrame;
	class Map;

	class Optimizer {
	public:
		void static BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP,
			int nIterations = 5, bool *pbStopFlag = NULL, const unsigned long nLoopKF = 0,
			const bool bRobust = true);
		void static GlobalBundleAdjustemnt(Map* pMap, int nIterations = 5, bool *pbStopFlag = NULL,
			const unsigned long nLoopKF = 0, const bool bRobust = true);
		void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap);
		int static PoseOptimization(Frame* pFrame);
	};
}
#endif