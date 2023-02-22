#ifndef EDGE_SLAM_H
#define EDGE_SLAM_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <DBoW3.h>
#include <ThreadPool.h>
#include <Evaluation.h>
#include <ConcurrentMap.h>
#include <atomic>
//#include <ORBVocabulary.h>

namespace EdgeSLAM {

	class Initializer;
	class Tracker;
	class FeatureTracker;
	class LocalMapper;
	class LoopCloser;
	class User;
	class Map;
	class Visualizer;
	class SLAM {
	public:
		SLAM();
		SLAM(ThreadPool::ThreadPool* _pool);
		virtual ~SLAM();
	public:
		void Init();
		void Track(int id, std::string user, double ts = 0.0);
		void LoadVocabulary();
		void InitVisualizer(std::string user,std::string name, int w, int h);
		void ProcessContentGeneration(std::string user, int id);
		void ProcessSegmentation(std::string user, int id);
		void ProcessObjectDetection(std::string user, int id);
		void ProcessDepthEstimation(std::string user, int id);
	public:
		ThreadPool::ThreadPool* pool;
		Initializer* mpInitializer;
		Tracker* mpTracker;
		LocalMapper* mpLocalMapper;
		LoopCloser* mpLoopCloser;
		FeatureTracker* mpFeatureTracker;
		Visualizer* mpVisualizer;
		DBoW3::Vocabulary* mpDBoWVoc;
		//ORBVocabulary* mpDBoWVoc;
	public:
		std::thread* mptVisualizer;
	private:

	/////Multi user and multi map
	public:
		void CreateMap(std::string name, int nq);
		void CreateUser(std::string _user, std::string _map, int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, float _d5, int quality, int nskip, bool _b, bool _bTracking = false, bool _bIMU = false, bool _bGBA = false, bool _bsave = false, bool _basync = false);
		bool CheckMap(std::string str);
		bool CheckUser(std::string str);
		int  CountUser();
		void AddUser(std::string id, User* user);
		void RemoveUser(std::string id);
		User* GetUser(std::string id);
		std::vector<User*> GetAllUsersInMap(std::string map);
		void UpdateDeviceGyroSensor(std::string user, int id);
		void UpdateDevicePosition(std::string user, int id, double ts);
		void AddMap(std::string name, Map* pMap);
		Map* GetMap(std::string name);
		void RemoveMap(std::string name);

		void VisualizeImage(cv::Mat src, int vid);

		ConcurrentMap<std::string, User*> Users;
		ConcurrentMap<std::string, Map*> Maps;
		ConcurrentMap<std::string, int> MapQuality;
		ConcurrentMap<std::string, std::vector<cv::Mat>> TemporalDatas;
		ConcurrentMap<std::string, std::map<int, cv::Mat>> TemporalDatas2;

	private:
		/*std::mutex mMutexUserList, mMutexMapList;
		std::map<std::string, User*> mmpConnectedUserList;
		std::map<std::string, Map*> mmpMapList;*/
	////Manage Visualize ID
	public:
		int GetConnectedDevice();
		void SetUserVisID(User* user);
		void UpdateUserVisID();
	private:
		std::mutex mMutexVisID;
		int mnVisID;
		std::map<User*, int> mapVisID;

	//////Save Data
	public:
		ConcurrentMap<std::string, std::map<int, ProcessTime*>> ProcessingTime;
		ConcurrentMap<std::string, std::map<int, Ratio*>> SuccessRatio;
		//std::map<int, std::vector<ProcessTime*>> ProcessTime;
		/*
		int nTotalTrack, nTotalReloc, nTotalMapping, nTotalSeg, nTotalDepth;
		float nAvgTrack, nAvgReloc, nAvgMapping,  nAvgSeg, nAvgDepth;
		float nStdDevTrack, nStdDevReloc, nStdDevMapping, nStdDevSeg, nStdDevDepth;
		float fSumTrack, fSumTrack2, fSumMapping, fSumMapping2, fSumReloc, fSumReloc2, fSumSeg, fSumSeg2, fSumDepth, fSumDepth2;
		void UpdateTrackingTime(float ts);
		void UpdateRelocTime(float ts);
		void UpdateMappingTime(float ts);
		*/
		void InitProcessingTime();
		void SaveProcessingTime();
		void LoadProcessingTime();
		void SaveTrajectory(std::string user);
	public:
		ConcurrentMap<int, std::chrono::high_resolution_clock::time_point> RequestTime;
	private:
		std::mutex mMutexTrackingTime, mMutexRelocTime, mMutexMappingTime;
	};
}

#endif
