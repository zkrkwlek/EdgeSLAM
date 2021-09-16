#ifndef EDGE_SLAM_MAP_H
#define EDGE_SLAM_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <DBoW3.h>
//#include <ORBVocabulary.h>
#include <mutex>
#include <atomic>
#include <LoopClosingTypes.h>

namespace EdgeSLAM {
	
	enum class MapState {
		NoImages, NotInitialized, Initialized, NeedNewKeyFrame
	};
	class MapPoint;
	class TrackPoint;
	class KeyFrame;
	class Frame;
	class LocalMap;
	class KeyFrameDB;
	class User;
	class Map {
	/*public:
		typedef std::pair<std::set<KeyFrame*>, int> ConsistentGroup;*/
	public:
		Map(DBoW3::Vocabulary* voc, bool bFixScale = false);
		//Map(ORBVocabulary* voc);
		virtual ~Map();

	public:
		void AddMapPoint(MapPoint* pMP);
		void RemoveMapPoint(MapPoint* pMP);
		std::vector<MapPoint*> GetAllMapPoints();
		int GetNumMapPoints();

		void AddKeyFrame(KeyFrame* pF);
		KeyFrame* GetKeyFrame(int id);
		void RemoveKeyFrame(KeyFrame* pF);
		std::vector<KeyFrame*> GetAllKeyFrames();
		int GetNumKeyFrames();
		void Delete();

		void InformNewBigChange();
		int GetLastBigChangeIdx();

	public:
		std::vector<KeyFrame*> mvpKeyFrameOrigins;
		DBoW3::Vocabulary* mpVoc;
		bool mbAbortBA;
		std::atomic<int> mnNextKeyFrameID, mnNextMapPointID, mnMaxKFid, mnBigChangeIdx;
		KeyFrameDB* mpKeyFrameDB;
		
	public:
		////Local Mapper thread
		void InterruptBA() {
			mbAbortBA = true;
		}
		void RequestStop();
		bool stopRequested();
		bool Stop();
		bool isStopped();
		void Release();
		bool SetNotStop(bool flag);

		void RequestReset();
		void ResetIfRequested();

		bool CheckFinish();
		void SetFinish();
		void RequestFinish();
		bool isFinished();

		std::atomic<bool> mbResetRequested;
		std::atomic<bool> mbFinishRequested;
		std::atomic<bool> mbFinished;
		std::atomic<bool> mbStopped;
		std::atomic<bool> mbStopRequested;
		std::atomic<bool> mbNotStop;

	public:
		////Loop Closing
		bool isRunningGBA();
		bool isFinishedGBA();

		KeyFrame* mpMatchedKF;
		std::vector<ConsistentGroup> mvConsistentGroups;
		std::vector<KeyFrame*> mvpEnoughConsistentCandidates;
		std::vector<KeyFrame*> mvpCurrentConnectedKFs;
		std::vector<MapPoint*> mvpCurrentMatchedPoints;
		std::vector<MapPoint*> mvpLoopMapPoints;
		cv::Mat mScw;
		g2o::Sim3 mg2oScw;

		std::atomic<int> mnLastLoopKFid;
		std::atomic<bool> mbRunningGBA;
		std::atomic<bool> mbFinishedGBA;
		bool mbStopGBA; //for optimization
		std::mutex mMutexGBA;
		std::thread* mpThreadGBA;
		// Fix scale in the stereo/RGB-D case
		bool mbFixScale;
		int mnFullBAIdx;
		////Loop Closing

	private:
		std::mutex mMutexMPs, mMutexKFs, mMutexLocalMap;
		std::set<MapPoint*> mspMapPoints;
		std::map<int, KeyFrame*> mmpKeyFrames;
		LocalMap* mpLocalMap;
		////local map??
		////Plane Test
	public:
		std::vector<cv::Mat> GetPlanarMPs(int id);
		void ClearPlanarMPs();
		void AddPlanarMP(cv::Mat m, int id);
	private:
		std::mutex mMutexPlanarMP;
		std::vector<std::vector<cv::Mat>> mvvPlanarMPs;
		////Plane Test
		////Depth Test
	public:
		std::vector<cv::Mat> GetDepthMPs();
		void ClearDepthMPs();
		void AddDepthMP(cv::Mat m);
	private:
		std::mutex mMutexDepthTest;
		std::vector<cv::Mat> mvTempMPs;
	////Depth Test
	public:
		MapState GetState();
		void SetState(MapState stat);
		std::atomic<int> mnNumMappingFrames, mnNumLoopClosingFrames, mnNumPlaneEstimation;
		std::list<MapPoint*> mlpNewMPs;
	private:
		MapState mState;
		std::mutex mMutexState;
	public:
		std::mutex mMutexMapUpdate;
	};
	
	class LocalMap {
	public:
		LocalMap();
		virtual ~LocalMap();
	public:
		virtual void UpdateLocalMap(User* user, Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs, std::vector<TrackPoint*>& vpLocalTPs) {}
		virtual void UpdateLocalKeyFrames(User* user, Frame* f, std::vector<KeyFrame*>& vpLocalKFs) {}
		virtual void UpdateLocalMapPoitns(Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs, std::vector<TrackPoint*>& vpLocalTPs) {}

		virtual void UpdateLocalMap(KeyFrame* kf, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs) {}
		virtual void UpdateLocalKeyFrames(KeyFrame* kf, std::vector<KeyFrame*>& vpLocalKFs){}
		virtual void UpdateLocalMapPoitns(KeyFrame* kf, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs){}
	public:
		/*std::vector<MapPoint*> mvpLocalMapPoints;
		std::vector<KeyFrame*> mvpLocalKeyFrames;*/
	private:
		std::mutex mMutexLocalMap;
	};

	class LocalCovisibilityMap :public LocalMap {
	public:
		LocalCovisibilityMap();
		virtual ~LocalCovisibilityMap();
	public:
		void UpdateLocalMap(User* user, Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs, std::vector<TrackPoint*>& vpLocalTPs);
		void UpdateLocalKeyFrames(User* user, Frame* f, std::vector<KeyFrame*>& vpLocalKFs);
		void UpdateLocalMapPoitns(Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs, std::vector<TrackPoint*>& vpLocalTPs);
		//void UpdateKeyFrames
		void UpdateLocalMap(KeyFrame* kf, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs);
		void UpdateLocalKeyFrames(KeyFrame* kf, std::vector<KeyFrame*>& vpLocalKFs);
		void UpdateLocalMapPoitns(KeyFrame* kf, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs);
	private:

		/*
		def update_keyframes(self, kf_ref):
		with self._lock:
		assert(kf_ref is not None)
		self.keyframes = OrderedSet()
		self.keyframes.add(kf_ref)
		neighbor_kfs = [kf for kf in kf_ref.get_covisible_keyframes() if not kf.is_bad]
		self.keyframes.update(neighbor_kfs)
		return self.keyframes

		def get_best_neighbors(self, kf_ref, N=Parameters.kLocalMappingNumNeighborKeyFrames):
		return kf_ref.get_best_covisible_keyframes(N)

		# update the local keyframes, the viewed points and the reference keyframes (that see the viewed points but are not in the local keyframes)
		def update(self, kf_ref):
		self.update_keyframes(kf_ref)
		return self.update_from_keyframes(self.keyframes)
		*/

	};
}

/*
lock and update lock, update_lock is unused

frame.get_matched_good_points()
kf_ref.get_best_covisible_keyframes(N)
*/

#endif

