#ifndef EDGE_SLAM_FRAME_H
#define EDGE_SLAM_FRAME_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <DBoW3.h>
//#include "DBoW2/BowVector.h"
//#include "DBoW2/FeatureVector.h"
//#include <ORBVocabulary.h>
#include <mutex>

namespace EdgeSLAM {

	const unsigned char FLAG_IMG = 0x1;
	const unsigned char FLAG_KP = 0x2;
	const unsigned char FLAG_DESC = 0x4;
	const unsigned char FLAG_SEG = 0x8;
	const unsigned char FLAG_DEPTH = 0x8;

	class Camera;
	class CameraPose;
	class FeatureDetector;
	class FeatureTracker;
	class MapPoint;
	class Frame {
	public:
		Frame();
		Frame(const Frame &src);
		Frame(cv::Mat img, Camera* pCam, int id, double time_stamp = 0.0);
		virtual ~Frame();
	public:
		bool is_in_frustum(MapPoint* pMP, float viewingCosLimit);
		bool is_in_image(float x, float y, float z = 1.0);
		void reset_map_points();
		void check_replaced_map_points();
	public:
		int mnKeyFrameId;
		int N;
		cv::Mat K, D;
		float fx, fy, cx, cy, invfx, invfy;
		bool mbDistorted;
		int FRAME_GRID_COLS;
		int FRAME_GRID_ROWS;
		float mfGridElementWidthInv;
		float mfGridElementHeightInv;
		std::vector<std::size_t> **mGrid;
		
		int mnScaleLevels;
		float mfScaleFactor;
		float mfLogScaleFactor;
		std::vector<float> mvScaleFactors;
		std::vector<float> mvInvScaleFactors;
		std::vector<float> mvLevelSigma2;
		std::vector<float> mvInvLevelSigma2;

		float mnMinX;
		float mnMaxX;
		float mnMinY;
		float mnMaxY;

	public:
		cv::Mat imgColor, imgGray;
		static FeatureDetector* detector;
		static FeatureTracker* matcher;
		CameraPose* mpCamPose;
		double mdTimeStamp;

		int mnFrameID;
		std::vector<cv::KeyPoint> mvKeys;
		std::vector<cv::KeyPoint> mvKeysUn;
		std::vector<MapPoint*> mvpMapPoints;
		std::vector<bool> mvbOutliers;
		cv::Mat mDescriptors;
		////Ref Frame mpReferenceKF
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
		////implement pose function from CameraPose class
		Camera* mpCamera;
		void SetPose(const cv::Mat &Tcw);
		cv::Mat GetPose();
		cv::Mat GetPoseInverse();
		cv::Mat GetCameraCenter();
		cv::Mat GetRotation();
		cv::Mat GetTranslation();
		////implement function from Camera class, for example is in frame
	public:
/*
	Feature Grid Distribution	
*/
	public:
		std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel = -1, const int maxLevel = -1) const;
	private:
		void UndistortKeyPoints();
		void AssignFeaturesToGrid();
		bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);
	private:
		std::mutex mMutexFeatures;
	/////FLAG
	public:
		bool CheckFlag(unsigned char opt);
		void TurnOnFlag(unsigned char opt);
		void TurnOffFlag(unsigned char opt);
	private:
		int mnFlag;
		std::mutex mMutexFlag;
	/////FLAG
	};
}

#endif