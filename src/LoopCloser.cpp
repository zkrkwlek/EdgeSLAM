#include <LoopCloser.h>
#include <Map.h>
#include <Tracker.h>
#include <LocalMapper.h>
#include <KeyFrameDB.h>

namespace EdgeSLAM {
	LoopCloser::LoopCloser(DBoW3::Vocabulary* voc, bool bFixScale){}
	//LoopCloser::LoopCloser(ORBVocabulary* voc, bool bFixScale) {}
	LoopCloser::~LoopCloser(){}
	void LoopCloser::ProcessLoopClosing(SLAM* system, Map* map, KeyFrame* kf) {
		map->mpKeyFrameDB->add(kf);
	}
}