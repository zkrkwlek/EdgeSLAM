#ifndef EDGE_SLAM_LOOP_CLOSER_H
#define EDGE_SLAM_LOOP_CLOSER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <DBoW3.h>
//#include <ORBVocabulary.h>

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
		LoopCloser(DBoW3::Vocabulary* voc, bool bFixScale);
		//LoopCloser(ORBVocabulary* voc, bool bFixScale);
		virtual ~LoopCloser();
		static void ProcessLoopClosing(SLAM* system, Map* map, KeyFrame* kf);
	};
}
#endif