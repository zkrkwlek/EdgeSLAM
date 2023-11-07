#ifndef EDGE_SLAM_OBJECT_FRAME_H
#define EDGE_SLAM_OBJECT_FRAME_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ConcurrentVector.h>
#include <ConcurrentSet.h>
#include <ConcurrentMap.h>
#include <MapPoint.h>
#include <Node.h>
#include <DBoW3.h>

class KalmanFilter;

namespace EdgeSLAM {
	class FeatureTracker;
	class Frame;
	class KeyFrame;
	class MapPoint;
	class Map;
	class ObjectNode;
	class ObjectMapPoint;
	class ObjectLocalMap;
	class CameraPose;

	enum class ObjectTrackingState {
		NotEstimated, Success, Failed
	};
	class ObjectTrackingFrame {
	public:
		ObjectTrackingFrame();
		virtual~ObjectTrackingFrame();
	public:
		cv::Mat frame;
		std::vector<EdgeSLAM::MapPoint*> mvpMapPoints;
		std::vector<cv::Point2f> mvImagePoints;
	};
	class ObjectTrackingResult {
	public:
		ObjectTrackingResult(ObjectNode* _pObj, int _label, std::string _user);
		virtual ~ObjectTrackingResult();
	public:
		ObjectTrackingState mState;
		int mnObjectLabelId;
		std::string mStrDeviceName;
		int mnLastSuccessFrameId;
		int mnLastTrackFrameId;
		cv::Mat Pose;
		ObjectTrackingFrame* mpLastFrame;
		ObjectNode* mpObject;
	};

	class ObjectLocalMap {
	public:
		ObjectLocalMap(std::vector<MapPoint*> vpMPs);
		virtual ~ObjectLocalMap();
	public:
		std::vector<MapPoint*> mvpLocalMapPoints;
		std::vector<ObjectBoundingBox*> mvpLocalBoxes;
	};

	class ObjectBoundingBox {
	public:
		//ObjectBoundingBox();
		//ObjectBoundingBox(int _label, float _conf, cv::Point2f pt1, cv::Point2f pt2);
		ObjectBoundingBox(Frame* _pF, int _label, float _conf, cv::Point2f pt1, cv::Point2f pt2);
		ObjectBoundingBox(KeyFrame* _pKF, int _label, float _conf, cv::Point2f pt1, cv::Point2f pt2);
		virtual ~ObjectBoundingBox();
	public:
		void AddMapPoint(MapPoint* pMP, size_t idx);
		void AddMapPoint(MapPoint* pMP);
		void EraseObjectPointMatch(ObjectMapPoint* pMP);
		void EraseObjectPointMatch(const size_t& idx);
	public:
		float fx, fy, cx, cy;
	public:
		int id;
		int label;
		float confidence;
		cv::Rect rect;

		int N;
		cv::Mat K;
		cv::Mat desc;
		std::vector<cv::KeyPoint> mvKeys;
		std::vector<bool> mvbOutliers;
		ConcurrentVector<ObjectMapPoint*> mvpObjectPoints;
		ConcurrentVector<MapPoint*> mvpMapPoints;
		ConcurrentSet<MapPoint*> mspMapPoints;
		std::vector<int> mvIDXs; //연결 된 키프레임에서 인덱스 정보를 알기 위해
		std::map<int, int> mapIDXs; //인덱스 연결. 프레임 인덱스에서 박스 인덱스 얻기
		KeyFrame* mpKF;
		Frame* mpF;

		void ComputeBow(DBoW3::Vocabulary* voc);
		DBoW3::BowVector mBowVec;
		DBoW3::FeatureVector mFeatVec;

		CameraPose* Pose;
		ObjectNode* mpNode;
		//이 안에 디스크립터와 피쳐 쓰기?
	public:
		//scale
		const int mnScaleLevels;
		const float mfScaleFactor;
		const float mfLogScaleFactor;
		const std::vector<float> mvScaleFactors;
		const std::vector<float> mvLevelSigma2;
		const std::vector<float> mvInvLevelSigma2;
	public:
		void AddConnection(ObjectBoundingBox* pBox, const int& weight);
		void EraseConnection(ObjectBoundingBox* pBox);
		void UpdateConnections(ObjectNode* Object);
		void UpdateConnections();
		void UpdateBestCovisibles();
		std::set<ObjectBoundingBox*> GetConnectedBoxes();
		std::vector<ObjectBoundingBox* > GetVectorCovisibleBoxes();
		std::vector<ObjectBoundingBox*> GetBestCovisibilityBoxes(const int& N);
		std::vector<ObjectBoundingBox*> GetCovisiblesByWeight(const int& w);
		int GetWeight(ObjectBoundingBox* pBox);
		void AddChild(ObjectBoundingBox* pBox);
		void EraseChild(ObjectBoundingBox* pBox);
		void ChangeParent(ObjectBoundingBox* pBox);
		std::set<ObjectBoundingBox*> GetChilds();
		ObjectBoundingBox* GetParent();
		bool hasChild(ObjectBoundingBox* pBox);
		
		static bool weightComp(int a, int b) {
			return a > b;
		}

		static bool lId(ObjectBoundingBox* pKF1, ObjectBoundingBox* pKF2) {
			return pKF1->id < pKF2->id;
		}

	private:
		std::mutex mMutexConnections;
		ConcurrentMap<ObjectBoundingBox*, int> mConnectedBoxWeights;
		ConcurrentVector<ObjectBoundingBox*> mvpOrderedConnectedBoxes;
		ConcurrentVector<int> mvOrderedWeights;

		// Spanning Tree and Loop Edges

		bool mbFirstConnection;
		ObjectBoundingBox* mpParent;
		ConcurrentSet<ObjectBoundingBox*> mspChildrens;
	};

	class ObjectMapPoint {
	public:
		ObjectMapPoint(ObjectBoundingBox* pRefBB, MapPoint* pMP);
		virtual ~ObjectMapPoint();
		int GetIndexInKeyFrame(ObjectBoundingBox* pBox);
		void AddObservation(ObjectBoundingBox* pBB, size_t idx);
		void EraseObservation(ObjectBoundingBox* pBB);
		std::map<ObjectBoundingBox*, size_t> GetObservations();
		//void ComputeDistinctiveDescriptors();
		//void SetBadFlag();
		//cv::Mat GetObjectPos();
		//void SetObjectPos(const cv::Mat& _pos);
	public:
		MapPoint* mpMapPoint;
		ObjectNode* mpObjectMap;
	private:
		std::mutex mMutexObjectPos;
		cv::Mat matPosObject;
		std::atomic<int> nObs;
	protected:
		
		ObjectBoundingBox* mpRefBoudingBox;
		ConcurrentMap<ObjectBoundingBox*, size_t> mObservations;
	};

	class ObjectNode : Node{
	public:
		ObjectNode();
		virtual ~ObjectNode();
		int mnId;
		int mnLabel;
		float radius;
		float radius_x, radius_y, radius_z;
		cv::Mat origin;
	public:
		void RemoveMapPoint(ObjectMapPoint* pMP);
	public:
		KalmanFilter* mpKalmanFilter;
		void UpdateOrigin();
		void UpdateOrigin(std::vector<MapPoint*>& vecMPs);
		cv::Mat GetOrigin();

		void UpdateObjectPos();

		std::mutex mMutexOrigin;

		DBoW3::BowVector mBowVec;
		DBoW3::FeatureVector mFeatVec;
		
		std::list<ObjectMapPoint*> mlpNewMPs;

		ConcurrentSet<ObjectMapPoint*> mspOPs;
		ConcurrentSet<KeyFrame*> mspKFs;
		ConcurrentSet<ObjectBoundingBox*> mspBBs;

		ConcurrentSet<EdgeSLAM::MapPoint*> mspMPs;
		//인접한 바운딩 박스 연결 -  슬램처럼

		void ComputeBow(DBoW3::Vocabulary* voc);
		void ClearDescriptor();
		void AddDescriptor(cv::Mat _row);
		cv::Mat GetDescriptor();
	public:
		void SetWorldPose(const cv::Mat& _pos);
		void SetObjectPose(const cv::Mat& _pos);
		cv::Mat GetWorldPose();
		cv::Mat GetObjectPose();
	private:
		cv::Mat matObjPose, matWorldPose;
		std::mutex mMutexObjPose, mMutexWorldPose;

	private:
		std::mutex mMutexDesc;
		cv::Mat desc;
	};
	
	class ObjectFrame {
	public:
		ObjectFrame();
		virtual ~ObjectFrame();
	public:
		//std::map<int, ObjectBox*> mapObjects;
	private:
		
	};
}
#endif