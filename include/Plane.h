#ifndef EDGE_SLAM_PLANE_H
#define EDGE_SLAM_PLANE_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace EdgeSLAM {
	class SLAM;
	class MapPoint;
	class KeyFrame;
	class User;
	class Map;
	class Plane {
	public:
		Plane();
		virtual ~Plane();
	public:
		void SetParam(cv::Mat m);
		void SetParam(cv::Mat n, float d);
		void GetParam(cv::Mat& n, float& d);
		cv::Mat GetParam();
	public:
		std::vector<MapPoint*> mvpMPs;
		bool mbParallel, mbInit;
	private:
		std::mutex mMutexParam;
		cv::Mat normal;
		float distance;
		float norm;
		cv::Mat matPlaneParam;
	};

	//frame or keyframe id
	class LocalIndoorModel {
	public:
		LocalIndoorModel();
		LocalIndoorModel(KeyFrame* pKF);
		virtual ~LocalIndoorModel();
	public:
		KeyFrame* mpTargetKF;
		std::vector<KeyFrame*> mvpLocalKFs;
		std::vector<MapPoint*> mvpLocalMPs;
		Plane *mpFloor, *mpCeil, *mpWall1, *mpWall2, *mpWall3;
	private:

	};

	class PlaneProcessor {
	public:
		std::map<int, LocalIndoorModel*> LocalPlanarMap;
	public:
		static void EstimateLocalMapPlanes(SLAM* system, Map* map, KeyFrame* pKF);
		static bool calcUnitNormalVector(cv::Mat& X);
		static int GetNormalType(cv::Mat X);
		static bool PlaneInitialization(Plane* plane,std::vector<MapPoint*> vpPoints, std::vector<MapPoint*>& vpOutlierMPs, int ransac_trial = 1500, float thresh_distance = 0.05, float thresh_ratio = 0.1);
		static cv::Mat CalcInverPlaneParam(cv::Mat P, cv::Mat Tinv);
		static float CalculateDepth(cv::Mat Xcam, cv::Mat Pinv);
		static cv::Mat CreateWorldPoint(cv::Mat Xcam, cv::Mat Tinv, float depth);
	private:
		static bool PlaneInitialization2(cv::Mat src, cv::Mat& res, cv::Mat& matInliers, cv::Mat& matOutliers, int ransac_trial, float thresh_distance, float thresh_ratio);
		static bool Ransac_fitting(cv::Mat src, cv::Mat& res, cv::Mat& matInliers, cv::Mat& matOutliers, int ransac_trial, float thresh_distance, float thresh_ratio);
		static cv::Mat CalcPlaneRotationMatrix(cv::Mat normal);

		static void CreatePlanarMapPoints(Map* map, KeyFrame* targetKF, cv::Mat param);
	private:
		static float fHistSize;
		static int nTrial;
		static float fDistance;
		static float fRatio; 
		static float fNormal;
	};
}

#endif





