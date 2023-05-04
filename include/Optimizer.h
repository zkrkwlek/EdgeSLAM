#ifndef EDGE_SLAM_OPTIMIZER_H
#define EDGE_SLAM_OPTIMIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>
#include <LoopClosingTypes.h>
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/types/types_seven_dof_expmap.h"

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
		void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap, long long ts);
		int static PoseOptimization(Frame* pFrame);
		int static OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches1,
			g2o::Sim3 &g2oS12, const float th2, const bool bFixScale);
		void static OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
			const KeyFrameAndPose &NonCorrectedSim3,
			const KeyFrameAndPose &CorrectedSim3,
			const std::map<KeyFrame *, std::set<KeyFrame *> > &LoopConnections,
			const bool &bFixScale);
	};
}
#endif