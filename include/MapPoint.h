#ifndef EDGE_SLAM_MAP_POINT_H
#define EDGE_SLAM_MAP_POINT_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>
#include <ConcurrentSet.h>
#include <ConcurrentMap.h>
#include <atomic>

class Confidence;

namespace EdgeSLAM {
	class KeyFrame;
	class Frame;
	class Map;
	class FeatureTracker;
	class User;

	class SemanticLabel;
	class ObjectNode;
	class ObjectBoundingBox;
	class ObjectMapPoint;

	class TrackPoint {
	public:
		TrackPoint();
		TrackPoint(float x, float y, float angle, float scale);
		virtual ~TrackPoint();
	public:
		float mTrackProjX;
		float mTrackProjY;
		float mTrackProjXR;
		bool mbTrackInView;
		int mnTrackScaleLevel;
		float mTrackViewCos;
		long unsigned int mnTrackReferenceForFrame;
		long unsigned int mnLastFrameSeen;

	private:
	};
	class MapPoint
	{
	public:
		MapPoint(const cv::Mat &Pos, KeyFrame* pRefKF, Map* pMap, long long _ts);
		//MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF);
		virtual ~MapPoint();

		void SetWorldPos(const cv::Mat &Pos);
		cv::Mat GetWorldPos();

		cv::Mat GetNormal();
		KeyFrame* GetReferenceKeyFrame();

		//MAP
		std::map<KeyFrame*, size_t> GetObservations();
		int Observations();
		void AddObservation(KeyFrame* pKF, size_t idx);
		void EraseObservation(KeyFrame* pKF);
		int GetIndexInKeyFrame(KeyFrame* pKF);
		bool IsInKeyFrame(KeyFrame* pKF);
		//BOX
		
		int BoxObservations();
		void AddBoxObservation(ObjectBoundingBox* pBB, size_t idx);
		void EraseBoxObservation(ObjectBoundingBox* pBB);
		std::map<ObjectBoundingBox*, size_t> GetBoxObservations();
		int GetIndexInBox(ObjectBoundingBox* pBox);
		bool IsInBox(ObjectBoundingBox* pKF);

		void SetBadFlag();
		bool isBad();

		void Replace(MapPoint* pMP);
		MapPoint* GetReplaced();

		void IncreaseVisible(int n = 1);
		void IncreaseFound(int n = 1);
		float GetFoundRatio();
		inline int GetFound() {
			return mnFound;
		}

		void ComputeDistinctiveDescriptors();

		cv::Mat GetDescriptor();

		void UpdateNormalAndDepth();

		float GetMinDistanceInvariance();
		float GetMaxDistanceInvariance();
		int PredictScale(const float &currentDist, KeyFrame* pKF);
		int PredictScale(const float &currentDist, Frame* pF);

		ConcurrentSet<User*> mSetConnected;
		//ConcurrentSet<ObjectBoundingBox*> mspBBs;
		//ObjectNode* mpObjectNode;
	public:
		int mnId;
		int mnFirstKFid;
		int mnFirstFrame;
		int nObs;
		std::atomic<int> nBoxObs;
		std::atomic<int> mnObjectID;
		std::atomic<int> mnLabelID;
		std::atomic<int> mnPlaneID;
		std::atomic<int> mnPlaneCount;
		std::atomic<long long> mnLastUpdatedTime;

		SemanticLabel* mpSemanticLabel;
		ObjectMapPoint* mpObjectPoint;
		Confidence* mpConfidence;

		// Variables used by the tracking
		/*float mTrackProjX;
		float mTrackProjY;
		float mTrackProjXR;
		bool mbTrackInView;
		int mnTrackScaleLevel;
		float mTrackViewCos;
		long unsigned int mnTrackReferenceForFrame;
		long unsigned int mnLastFrameSeen;*/

		// Variables used by local mapping
		long unsigned int mnBALocalForKF;
		long unsigned int mnFuseCandidateForKF;

		// Variables used by loop closing
		long unsigned int mnLoopPointForKF;
		long unsigned int mnCorrectedByKF;
		long unsigned int mnCorrectedReference;
		cv::Mat mPosGBA;
		long unsigned int mnBAGlobalForKF;

		static std::mutex mGlobalMutex;
		static FeatureTracker* mpDist;
	protected:

		// Position in absolute coordinates
		cv::Mat mWorldPos;

		// Keyframes observing the point and associated index in keyframe
		std::map<KeyFrame*, size_t> mObservations;
		ConcurrentMap<ObjectBoundingBox*, size_t> mBoxObservations;

		// Mean viewing direction
		cv::Mat mNormalVector;

		// Best descriptor to fast matching
		cv::Mat mDescriptor;

		// Reference KeyFrame
		KeyFrame* mpRefKF;

		// Tracking counters
		int mnVisible;
		int mnFound;

		// Bad flag (we do not currently erase MapPoint from memory)
		bool mbBad;
		MapPoint* mpReplaced;

		// Scale invariance distances
		float mfMinDistance;
		float mfMaxDistance;

		Map* mpMap;

		std::mutex mMutexPos;
		std::mutex mMutexFeatures;
	};
}

#endif