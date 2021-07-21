#ifndef EDGE_SLAM_MAP_H
#define EDGE_SLAM_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <DBoW3.h>
//#include <ORBVocabulary.h>
#include <mutex>
#include <atomic>

namespace EdgeSLAM {
	
	enum class MapState {
		NoImages, NotInitialized, Initialized, NeedNewKeyFrame
	};
	class MapPoint;
	class KeyFrame;
	class Frame;
	class LocalMap;
	class KeyFrameDB;
	class User;
	class Map {
	public:
		Map(DBoW3::Vocabulary* voc);
		//Map(ORBVocabulary* voc);
		virtual ~Map();

	public:
		void AddMapPoint(MapPoint* pMP);
		void RemoveMapPoint(MapPoint* pMP);
		std::vector<MapPoint*> GetAllMapPoints();
		int GetNumMapPoints();

		void AddKeyFrame(KeyFrame* pF);
		void RemoveKeyFrame(KeyFrame* pF);
		std::vector<KeyFrame*> GetAllKeyFrames();
		int GetNumKeyFrames();
		void Delete();

		void InterruptBA() {
			mbAbortBA = true;
		}

	public:
		bool mbAbortBA;
		std::atomic<int> mnNextKeyFrameID, mnNextMapPointID;;
		KeyFrameDB* mpKeyFrameDB;
	private:
		std::mutex mMutexMPs, mMutexKFs, mMutexLocalMap;
		std::set<MapPoint*> mspMapPoints;
		std::set<KeyFrame*> mspKeyFrames;
		LocalMap* mpLocalMap;
		////local map??
	public:
		MapState GetState();
		void SetState(MapState stat);
		std::atomic<int> mnNumMappingFrames;
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
		virtual void UpdateLocalMap(User* user, Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs) {}
		virtual void UpdateLocalKeyFrames(User* user, Frame* f, std::vector<KeyFrame*>& vpLocalKFs) {}
		virtual void UpdateLocalMapPoitns(Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs) {}
	public:
		/*std::vector<MapPoint*> mvpLocalMapPoints;
		std::vector<KeyFrame*> mvpLocalKeyFrames;*/
	private:
		std::mutex mMutexLocalMap;

		/*
		def is_empty(self):
		with self._lock:
		return len(self.keyframes)==0

		def get_points(self):
		with self._lock:
		return self.points.copy()

		def num_points(self):
		with self._lock:
		return len(self.points)

		def get_keyframes(self):
		with self._lock:
		return self.keyframes.copy()

		def num_keyframes(self):
		with self._lock:
		return len(self.keyframes)

		# given some input local keyframes, get all the viewed points and all the reference keyframes (that see the viewed points but are not in the local keyframes)
		def update_from_keyframes(self, local_keyframes):
		local_keyframes = set([kf for kf in local_keyframes if not kf.is_bad])   # remove possible bad keyframes
		ref_keyframes = set()   # reference keyframes: keyframes not in local_keyframes that see points observed in local_keyframes

		good_points = set([p for kf in local_keyframes for p in kf.get_matched_good_points()])  # all good points in local_keyframes (only one instance per point)
		for p in good_points:
		# get the keyframes viewing p but not in local_keyframes
		for kf_viewing_p in p.keyframes():
		if (not kf_viewing_p.is_bad) and (not kf_viewing_p in local_keyframes):
		ref_keyframes.add(kf_viewing_p)
		# debugging stuff
		# if not any([f in local_frames for f in p.keyframes()]):
		#     Printer.red('point %d without a viewing keyframe in input frames!!' %(p.id))
		#     Printer.red('         keyframes: ',p.observations_string())
		#     for f in local_frames:
		#         if p in f.get_points():
		#             Printer.red('point {} in keyframe {}-{} '.format(p.id,f.id,list(np.where(f.get_points() is p)[0])))
		#     assert(False)

		with self.lock:
		#local_keyframes = sorted(local_keyframes, key=lambda x:x.id)
		#ref_keyframes = sorted(ref_keyframes, key=lambda x:x.id)
		self.keyframes = local_keyframes
		self.points = good_points
		self.ref_keyframes = ref_keyframes
		return local_keyframes, good_points, ref_keyframes


		# from a given input frame compute:
		# - the reference keyframe (the keyframe that sees most map points of the frame)
		# - the local keyframes
		# - the local points
		def get_frame_covisibles(self, frame):
		points = frame.get_matched_good_points()
		#keyframes = self.get_local_keyframes()
		#assert len(points) > 0
		if len(points) == 0:
		Printer.red('get_frame_covisibles - frame with not points')

		# for all map points in frame check in which other keyframes are they seen
		# increase counter for those keyframes
		viewing_keyframes = [kf for p in points for kf in p.keyframes() if not kf.is_bad]# if kf in keyframes]
		viewing_keyframes = Counter(viewing_keyframes)
		kf_ref = viewing_keyframes.most_common(1)[0][0]
		#local_keyframes = viewing_keyframes.keys()

		# include also some not-already-included keyframes that are neighbors to already-included keyframes
		for kf in list(viewing_keyframes.keys()):
		second_neighbors = kf.get_best_covisible_keyframes(Parameters.kNumBestCovisibilityKeyFrames)
		viewing_keyframes.update(second_neighbors)
		children = kf.get_children()
		viewing_keyframes.update(children)
		if len(viewing_keyframes) >= Parameters.kMaxNumOfKeyframesInLocalMap:
		break

		local_keyframes_counts = viewing_keyframes.most_common(Parameters.kMaxNumOfKeyframesInLocalMap)
		local_points = set()
		local_keyframes = []
		for kf,c in local_keyframes_counts:
		local_points.update(kf.get_matched_points())
		local_keyframes.append(kf)
		return kf_ref, local_keyframes, local_points
		*/
	};

	class LocalCovisibilityMap :public LocalMap {
	public:
		LocalCovisibilityMap();
		virtual ~LocalCovisibilityMap();
	public:
		void UpdateLocalMap(User* user, Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs);
		void UpdateLocalKeyFrames(User* user, Frame* f, std::vector<KeyFrame*>& vpLocalKFs);
		void UpdateLocalMapPoitns(Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs);
		//void UpdateKeyFrames
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

