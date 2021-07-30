#ifndef EDGE_SLAM_LOOP_CLOSER_H
#define EDGE_SLAM_LOOP_CLOSER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <DBoW3.h>
#include <LoopClosingTypes.h>
#include <mutex>

namespace EdgeSLAM {
	class SLAM;
	class KeyFrameDB;
	class KeyFrame;
	class Map;
	class Tracker;
	class LocalMapper;
	class LoopCloser {
	public:
		//using ConsistentGroup = typename Map::ConsistentGroup;
		/*typedef std::pair<std::set<KeyFrame*>, int> ConsistentGroup;
		typedef std::map<KeyFrame*, g2o::Sim3, std::less<KeyFrame*>,
			Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;*/
	public:
		LoopCloser();
		//LoopCloser(ORBVocabulary* voc, bool bFixScale);
		virtual ~LoopCloser();
		static void ProcessLoopClosing(SLAM* system, Map* map, KeyFrame* kf);
		bool DetectLoop(SLAM* system, Map* map, KeyFrame* kf);
		bool ComputeSim3(SLAM* system, Map* map, KeyFrame* kf);
		void CorrectLoop(SLAM* system, Map* map, KeyFrame* kf);
		void SearchAndFuse(Map* map, const KeyFrameAndPose &CorrectedPosesMap);
		void RunGlobalBundleAdjustment(SLAM* system, Map* map, KeyFrame* kf, int nLoopKF);
	public:
		int mnCovisibilityConsistencyTh;
	};
}
#endif