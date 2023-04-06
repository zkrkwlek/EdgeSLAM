#ifndef EDGE_SLAM_LOCALMAPPER_H
#define EDGE_SLAM_LOCALMAPPER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ThreadPool.h>
#include <atomic>
#include <mutex>

namespace EdgeSLAM {
	class SLAM;
	class Map;
	class MapPoint;
	class KeyFrame;
	class Frame;
	class LocalMapper {
	public:
		LocalMapper();
		virtual ~LocalMapper();
	public:
		static void ProcessMapping(ThreadPool::ThreadPool* pool, SLAM* system, Map* map, KeyFrame* targetKF);
		static void SendLocalMap(KeyFrame* targetKF);
		static void SendKeyFrameInformation(SLAM* system, std::string name, Map* map, KeyFrame* targetKF);
		void ProcessNewKeyFrame(Map* map, KeyFrame* targetKF);
		void MapPointCulling(Map* map, KeyFrame* targetKF);
		void CreateNewMapPoints(Map* map, KeyFrame* targetKF);
		void SearchInNeighbors(Map* map, KeyFrame* targetKF);
		void KeyFrameCulling(Map* map, KeyFrame* targetKF);
		
		// Thread Synch
		/*void RequestStop();
		bool stopRequested();
		bool Stop();
		bool isStopped();
		void Release();
		void InterruptBA();
		bool SetNotStop(bool flag);
		
		void RequestReset();
		void ResetIfRequested();
		
		bool CheckFinish();
		void SetFinish();
		void RequestFinish();
		bool isFinished();*/
		
	public:
		std::mutex mMutexStop;
		std::mutex mMutexReset;
		std::mutex mMutexFinish;
		std::mutex mMutexNewKFs;
		bool mbResetRequested;
		bool mbFinishRequested;
		bool mbFinished;
		bool mbAbortBA;
		bool mbStopped;
		bool mbStopRequested;
		bool mbNotStop;
	private:

	};
}

#endif