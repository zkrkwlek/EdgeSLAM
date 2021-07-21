#ifndef EDGE_SLAM_KEYFRAME_DB_H
#define EDGE_SLAM_KEYFRAME_DB_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <DBoW3.h>
//#include <ORBVocabulary.h>

namespace EdgeSLAM {
	class KeyFrame;
	class Frame;
	class KeyFrameDB {
	public:
		KeyFrameDB(DBoW3::Vocabulary *voc);
		//KeyFrameDB(ORBVocabulary *voc);
		virtual ~KeyFrameDB();
		void add(KeyFrame* pKF);

		void erase(KeyFrame* pKF);

		void clear();

		// Loop Detection
		std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);

		// Relocalization
		std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F);

	protected:

		// Associated vocabulary
		DBoW3::Vocabulary* mpVoc;
		//ORBVocabulary* mpVoc;

		// Inverted file
		std::vector<std::list<KeyFrame*>> mvInvertedFile;

		// Mutex
		std::mutex mMutex;
	};
}
#endif