#ifndef EDGE_SLAM_INITIALIZER_H
#define EDGE_SLAM_INITIALIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <stack>

namespace EdgeSLAM {
	enum class MapState;
	class Frame;
	class KeyFrame;
	class Map;
	class FeatureTracker;
	class Initializer {
	public:
		Initializer(int nMinFeatures = 100, int nMinTriangulatedPoints = 100, int nMaxIdDistBetweenIntializingFrames = 10, int nNumOfFailuresAfterWichNumMinTriangulatedPointsIsHalved = 20);
		virtual ~Initializer();
	public:
		int mnMinFeatures, mnMinTriangulatedPoints, mnNumFailures, mnMaxIdDistBetweenIntializingFrames, mnNumOfFailuresAfterWichNumMinTriangulatedPointsIsHalved;
		
		FeatureTracker* mpFeatureTracker;
		Frame* mpRef;
		KeyFrame *mpInitKeyFrame1, *mpInitKeyFrame2;
		std::stack<Frame*> mFrameStack;
		/*
		 self.num_min_features = Parameters.kInitializerNumMinFeatures
        self.num_min_triangulated_points = Parameters.kInitializerNumMinTriangulatedPoints       
        self.num_failures = 0 
		*/
		void Reset();
		void Init(Frame* pRef);
		MapState Initialize(Frame* pCur, Map* pMap);
		void EstimatePose();
	private:
		void ReplaceReferenceFrame();
	};
}

#endif