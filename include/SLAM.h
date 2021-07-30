#ifndef EDGE_SLAM_H
#define EDGE_SLAM_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <DBoW3.h>
#include <ThreadPool.h>
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
		virtual ~SLAM();
	public:
		void Init();
		void Track(cv::Mat im, int id,User* user, double ts = 0.0);
		void LoadVocabulary();
		void InitVisualizer(std::string user,std::string name, int w, int h);
		void ProcessSegmentation(std::string user, int id);
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
		void CreateMap(std::string name);
		void CreateUser(std::string _user, std::string _map, int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, bool _b);
		bool CheckMap(std::string str);
		bool CheckUser(std::string str);
		void AddUser(std::string id, User* user);
		User* GetUser(std::string id);
		void RemoveUser(std::string id);
		void AddMap(std::string name, Map* pMap);
		Map* GetMap(std::string name);
		void RemoveMap(std::string name);
	private:
		std::mutex mMutexUserList, mMutexMapList;
		std::map<std::string, User*> mmpConnectedUserList;
		std::map<std::string, Map*> mmpMapList;
	};
}

#endif
