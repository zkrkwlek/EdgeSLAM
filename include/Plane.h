#ifndef EDGE_SLAM_PLANE_H
#define EDGE_SLAM_PLANE_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace EdgeSLAM {
	class MapPoint;
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

	class PlaneProcessor {
	public:
		static bool calcUnitNormalVector(cv::Mat& X);
		static int GetNormalType(cv::Mat X);
		static bool PlaneInitialization(Plane* plane,std::vector<MapPoint*> vpPoints, std::vector<MapPoint*>& vpOutlierMPs, int ransac_trial = 1500, float thresh_distance = 0.05, float thresh_ratio = 0.1);
		static cv::Mat CalcInverPlaneParam(cv::Mat P, cv::Mat Tinv);
		static float CalculateDepth(cv::Mat Xcam, cv::Mat Pinv);
		static cv::Mat CreateWorldPoint(cv::Mat Xcam, cv::Mat Tinv, float depth);
	};
}

#endif





