#ifndef EDGE_SLAM_KEYFRAME_H
#define EDGE_SLAM_KEYFRAME_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <DBoW3.h>
//#include "DBoW2/BowVector.h"
//#include "DBoW2/FeatureVector.h"
//#include <ORBVocabulary.h>
#include <atomic>
#include <mutex>

namespace EdgeSLAM {
	class MapPoint;
	class Map;
	class Frame;
	class Camera;
	class CameraPose;
	class FeatureTracker;
	class KeyFrame {
	public:
		KeyFrame(Frame *F, Map* pMap);
		virtual ~KeyFrame();
	public:
		bool is_in_image(float x, float y, float z = 1.0);
		void reset_map_points();
	public:
		// Covisibility graph functions
		void AddConnection(KeyFrame* pKF, const int &weight);
		void EraseConnection(KeyFrame* pKF);
		void UpdateConnections();
		void UpdateBestCovisibles();
		std::set<KeyFrame *> GetConnectedKeyFrames();
		std::vector<KeyFrame* > GetVectorCovisibleKeyFrames();
		std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(const int &N);
		std::vector<KeyFrame*> GetCovisiblesByWeight(const int &w);
		int GetWeight(KeyFrame* pKF);
	public:
		////dbow
		static DBoW3::Vocabulary* mpVoc;
		DBoW3::BowVector mBowVec;
		DBoW3::FeatureVector mFeatVec;
		/*static ORBVocabulary* mpVoc;
		DBoW2::BowVector mBowVec;
		DBoW2::FeatureVector mFeatVec;*/
		void ComputeBoW();
	public:
		// Spanning tree functions
		void AddChild(KeyFrame* pKF);
		void EraseChild(KeyFrame* pKF);
		void ChangeParent(KeyFrame* pKF);
		std::set<KeyFrame*> GetChilds();
		KeyFrame* GetParent();
		bool hasChild(KeyFrame* pKF);
	public:
		// Loop Edges
		void AddLoopEdge(KeyFrame* pKF);
		std::set<KeyFrame*> GetLoopEdges();
	public:
		// MapPoint observation functions
		void AddMapPoint(MapPoint* pMP, const size_t &idx);
		void EraseMapPointMatch(const size_t &idx);
		void EraseMapPointMatch(MapPoint* pMP);
		void ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP);
		std::set<MapPoint*> GetMapPoints();
		std::vector<MapPoint*> GetMapPointMatches();
		int TrackedMapPoints(const int &minObs);
		MapPoint* GetMapPoint(const size_t &idx);
		std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
		float ComputeSceneMedianDepth(const int q);
	public:
		// Enable/Disable bad flag changes
		void SetNotErase();
		void SetErase();
		// Set/check bad flag
		void SetBadFlag();
		bool isBad();

		static bool weightComp(int a, int b) {
			return a>b;
		}

		static bool lId(KeyFrame* pKF1, KeyFrame* pKF2) {
			return pKF1->mnId<pKF2->mnId;
		}
	
	public:
		Camera* mpCamera;
		CameraPose* mpCamPose;
		void SetPose(const cv::Mat &Tcw);
		cv::Mat GetPose();
		cv::Mat GetPoseInverse();
		cv::Mat GetCameraCenter();
		cv::Mat GetRotation();
		cv::Mat GetTranslation();

	public:
		static FeatureTracker* matcher;
		int mnId;
		const int mnFrameId;
		const double mTimeStamp;

		// Grid (to speed up feature matching)
		const int mnGridCols;
		const int mnGridRows;
		const float mfGridElementWidthInv;
		const float mfGridElementHeightInv;

		// Variables used by the tracking
		//long unsigned int mnTrackReferenceForFrame;
		long unsigned int mnFuseTargetForKF;

		// Variables used by the local mapping
		long unsigned int mnBALocalForKF;
		long unsigned int mnBAFixedForKF;

		// Variables used by the keyframe database
		long unsigned int mnLoopQuery;
		int mnLoopWords;
		float mLoopScore;
		long unsigned int mnRelocQuery;
		int mnRelocWords;
		float mRelocScore;

		// Variables used by loop closing
		cv::Mat mTcwGBA;
		cv::Mat mTcwBefGBA;
		long unsigned int mnBAGlobalForKF;

		// Calibration parameters
		const float fx, fy, cx, cy, invfx, invfy;// , mbf, mb, mThDepth;

		// Number of KeyPoints
		const int N;

		// KeyPoints, stereo coordinate and descriptors (all associated by an index)
		const std::vector<cv::KeyPoint> mvKeys;
		const std::vector<cv::KeyPoint> mvKeysUn;
		const cv::Mat mDescriptors;

		////BoW
		//DBoW2::BowVector mBowVec;
		//DBoW2::FeatureVector mFeatVec;

		// Pose relative to parent (this is computed when bad flag is activated)
		cv::Mat mTcp;

		// Scale
		const int mnScaleLevels;
		const float mfScaleFactor;
		const float mfLogScaleFactor;
		const std::vector<float> mvScaleFactors;
		const std::vector<float> mvLevelSigma2;
		const std::vector<float> mvInvLevelSigma2;

		// Image bounds and calibration
		const int mnMinX;
		const int mnMinY;
		const int mnMaxX;
		const int mnMaxY;
		const cv::Mat K;

		std::vector<bool> mvbOutliers;
		std::vector<MapPoint*> mvpMapPoints;
		std::vector< std::vector <std::vector<size_t> > > mGrid;

		std::map<KeyFrame*, int> mConnectedKeyFrameWeights;
		std::vector<KeyFrame*> mvpOrderedConnectedKeyFrames;
		std::vector<int> mvOrderedWeights;

		// Spanning Tree and Loop Edges
		bool mbFirstConnection;
		KeyFrame* mpParent;
		std::set<KeyFrame*> mspChildrens;
		std::set<KeyFrame*> mspLoopEdges;

		// Bad flags
		bool mbNotErase;
		bool mbToBeErased;
		bool mbBad;

		//float mHalfBaseline; // Only for visualization

		Map* mpMap;

		std::mutex mMutexPose;
		std::mutex mMutexConnections;
		std::mutex mMutexFeatures;
	};
}

#endif

/*
# ===============================
# spanning tree
def add_child(self, keyframe):

def erase_child(self, keyframe):

def set_parent(self, keyframe):

def get_children(self):

def get_parent(self):

def has_child(self, keyframe):


# ===============================
# loop edges
def add_loop_edge(self, keyframe):

def get_loop_edges(self):

# ===============================
# covisibility

def reset_covisibility(self):

def add_connection(self, keyframe, weigth):

def erase_connection(self, keyframe):

def update_best_covisibles(self):

# get a list of all the keyframe that shares points
def get_connected_keyframes(self):

# get an ordered list of covisible keyframes
def get_covisible_keyframes(self):

# get an ordered list of covisible keyframes
def get_best_covisible_keyframes(self,N):

def get_covisible_by_weight(self,weight):

def get_weight(self,keyframe):

def is_bad(self):

def set_not_erase(self):

def set_erase(self):

def set_bad(self):

##Frame function

*/