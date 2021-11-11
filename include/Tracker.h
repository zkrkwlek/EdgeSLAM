                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           #ifndef EDGE_SLAM_TRACKER_H
#define EDGE_SLAM_TRACKER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ThreadPool.h>

namespace EdgeSLAM {
	class User;
	class SLAM;
	class Frame;
	class KeyFrame;
	class MapPoint;
	class TrackPoint;
	class LocalMapper;
	class Map;
	class LocalMap;
	class Tracker {
	public:
		Tracker();
		virtual ~Tracker();
	public:
		static void UpdateDeviceGyro(SLAM* system, User* user, int id);
		static void Track(ThreadPool::ThreadPool* pool, SLAM* system, int id, User* user, double ts);
		static void SendTrackingResults(SLAM* system, User* user, int nFrameID, int n, cv::Mat R, cv::Mat t);
		static void SendDeviceTrackingData(SLAM* system, User* user, LocalMap* pLocalMap, Frame* frame, int nInlier, int id);
		bool TrackWithPrevFrame(Frame* prev, Frame* cur, float thMaxDesc, float thMinDesc);
		int TrackWithLocalMap(LocalMap* localMap, User* user, Frame* cur, float thMaxDesc, float thMinDesc);
		bool TrackWithKeyFrame(KeyFrame* ref, Frame* cur);
		int Relocalization(Map* map, User* user, Frame* cur, float thMinDesc);
	private:
		int UpdateVisiblePoints(Frame* cur, std::vector<MapPoint*> vpLocalMPs, std::vector<TrackPoint*> vpLocalTPs);
		int UpdateFoundPoints(Frame* cur, bool bOnlyTracking = false);
		bool NeedNewKeyFrame(Map* map, LocalMapper* mapper, Frame* cur, KeyFrame* ref, int nMatchesInliers, int nLastKeyFrameId, int nLastRelocFrameID, bool bOnlyTracking = false, int nMaxFrames = 30, int nMinFrames = 0);
		void CreateNewKeyFrame(ThreadPool::ThreadPool* pool, SLAM* system, Map* map, LocalMapper* mapper, Frame* cur, User* user);
	};
}

#endif