#ifndef EDGE_SLAM_SEGMENTATOR_H
#define EDGE_SLAM_SEGMENTATOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ThreadPool.h>
#include <WebAPI.h>


namespace EdgeSLAM {
	
	enum class ObjectLabel {
		WALL = 1,
		FLOOR = 4,
		CEIL = 6
	};

	class SLAM;
	class Frame;
	class User;
	class MapPoint;
	class Plane;

	
	class Object{
	public:
		Object();
		virtual ~Object();
		void Update(int label);
		int GetLabel();
		cv::Mat GetLabels();
		int GetCount(int label);
		
	private:
		int mnLabel;
		int mnCount;
		cv::Mat matLabels;
		std::mutex mMutexObject;
	};

	class Segmentator {
	public:
		Segmentator();
		virtual ~Segmentator();
		static void ProcessPlanarModeling(SLAM* system, User* user);
		static void ProcessSegmentation(ThreadPool::ThreadPool* pool, SLAM* system, std::string user,int id);
		static void ProcessDepthEstimation(ThreadPool::ThreadPool* pool, SLAM* system, std::string user, int id);
		static void RequestSegmentation(std::string user,int id);
		static void ProcessContentGeneration(SLAM* system, User* user, int id);
		static void ProcessDevicePosition(SLAM* system, User* user, int id);
		
		static void Init();
		
	public:
		static int mnMaxObjectLabel;
		static std::map<int, Object*> ObjectPoints;
		static std::set<MapPoint*> mspAllFloorPoints;
		static std::set<MapPoint*> mspAllWallPoints;
		static std::vector<cv::Vec3b> mvObjectLabelColors;

		//temp
		static std::atomic<int> mnContentID;
		static Plane* floorPlane;
		static Plane* wallPlane1;
		static Plane* wallPlane2;

		//////plane
		static cv::Mat CalcFlukerLine(cv::Mat P1, cv::Mat P2);
		static cv::Mat LineProjection(cv::Mat R, cv::Mat t, cv::Mat Lw1, cv::Mat K, float& m);
		static cv::Mat Lw1, Lw2;
	private:

	};
}

#endif