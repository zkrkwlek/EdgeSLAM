#include <Tracker.h>
#include <SLAM.h>
#include <Initializer.h>
#include <LocalMapper.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <KeyFrameDB.h>
#include <MapPoint.h>
#include <User.h>
#include <Map.h>
#include <Camera.h>
#include <MotionModel.h>
#include <FeatureTracker.h>
#include <SearchPoints.h>
#include <Optimizer.h>
#include <PnPSolver.h>
#include <Visualizer.h>

#include <chrono>
namespace EdgeSLAM {
	Tracker::Tracker(){}
	Tracker::~Tracker(){}

	void Tracker::Track(ThreadPool::ThreadPool* pool, SLAM* system, cv::Mat im, int id, User* user, double ts) {
		if (user->mbProgress)
			return;
		user->mbProgress = true;
		auto cam = user->mpCamera;
		auto map = user->mpMap;
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		Frame* frame = new Frame(im, cam, id, ts);
		user->mnCurrFrameID = frame->mnFrameID;

		std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);
		
		auto mapState = map->GetState();
		auto userState = user->GetState();
		auto trackState = UserState::NotEstimated;
		if (mapState == MapState::NoImages) {
			//set reference frame
			map->SetState(MapState::NotInitialized);
			system->mpInitializer->Init(frame);
		}
		if (mapState == MapState::NotInitialized) {
			//initialization
			auto res = system->mpInitializer->Initialize(frame, map);
			map->SetState(res);
			if (res == MapState::Initialized) {
				trackState = UserState::Success;
				auto kf1 = system->mpInitializer->mpInitKeyFrame1;
				auto kf2 = system->mpInitializer->mpInitKeyFrame2;
				user->mapKeyFrames[kf1->mnId] = kf1;
				user->mapKeyFrames[kf2->mnId] = kf2;
				user->mnReferenceKeyFrameID = kf2->mnId;
				user->mnLastKeyFrameID = frame->mnFrameID;
				pool->EnqueueJob(LocalMapper::ProcessMapping, pool, system, map, kf1);
				pool->EnqueueJob(LocalMapper::ProcessMapping, pool, system, map, kf2);
			}
		}
		int nInliers = 0;
		if (mapState == MapState::Initialized) {
			if (userState == UserState::NotEstimated) {
				//global localization
				//set reference keyframe and last keyframe
				user->mnLastRelocFrameId = frame->mnFrameID;

			}
			else {
				bool bTrack = false;
				if (userState == UserState::Success) {
					//std::cout << "Tracker::Start" << std::endl;
					auto f_ref = user->mapFrames[user->mnPrevFrameID];
					f_ref->check_replaced_map_points();
					cv::Mat Tpredict = user->PredictPose();
					frame->SetPose(Tpredict);
					//std::cout << "Tracker::PrevFrame" << std::endl;
					bTrack = system->mpTracker->TrackWithPrevFrame(f_ref, frame, system->mpFeatureTracker->max_descriptor_distance, system->mpFeatureTracker->min_descriptor_distance);
				}
				if (userState == UserState::Failed) {
					frame->reset_map_points();
					//std::cout << "Tracker::Relocalization" << std::endl;
					bTrack = system->mpTracker->Relocalization(map, user, frame, system->mpFeatureTracker->min_descriptor_distance);
					if (bTrack) {
						user->mnLastRelocFrameId = frame->mnFrameID;
					}
				}
				if (bTrack) {
					//std::cout << "Tracker::LocalMap" << std::endl;
					nInliers = system->mpTracker->TrackWithLocalMap(user, frame, system->mpFeatureTracker->max_descriptor_distance, system->mpFeatureTracker->min_descriptor_distance);
					if (frame->mnFrameID < user->mnLastRelocFrameId + 30 && nInliers < 50) {
						bTrack = false;
					}
					else if (nInliers < 30) {
						bTrack = false;
					}
					else {
						bTrack = true;
					}
				}
				if(!bTrack)
					trackState = UserState::Failed;
				if (bTrack)
					trackState = UserState::Success;
			}
			if (userState == UserState::Failed) {
				//subsection relocalization
				//pose failure handler
				//set reference keyframe and last keyframe
				//
			}
			if (userState == UserState::Success) {

				
			}
		}

		////update user frame
		user->SetState(trackState);
		if (trackState == UserState::Success) {
			//pose update
			user->UpdatePose(frame->GetPose());
			//check keyframe
			auto ref = user->mapKeyFrames[user->mnReferenceKeyFrameID];
			if (system->mpTracker->NeedNewKeyFrame(map, system->mpLocalMapper, frame, ref, nInliers, user->mnLastKeyFrameID.load(), user->mnLastRelocFrameId.load())) {
				system->mpTracker->CreateNewKeyFrame(pool, system, map, system->mpLocalMapper, frame, user);
			}
		}
		user->mapFrames[frame->mnFrameID] = frame;
		user->mnPrevFrameID = frame->mnFrameID;
		user->mbProgress = false;

		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		std::cout << id << ", " << frame->mvKeys.size() <<"="<<nInliers<< "=" << t_test1 << std::endl;

		if (mapState == MapState::Initialized && userState != UserState::NotEstimated) {
			for (int i = 0; i < frame->mvKeys.size(); i++) {
				auto pMP = frame->mvpMapPoints[i];
				cv::Scalar color = cv::Scalar(255, 0, 255);
				int r = 2;
				if (pMP && !pMP->isBad())
				{
					color.val[1] = 255;
					color.val[2] = 0;
					r++;
					cv::circle(im, frame->mvKeys[i].pt, r, color, -1);
				}
			}
			cv::Mat resized_test;
			cv::resize(im, resized_test, cv::Size(im.cols / 2, im.rows / 2));
			system->mpVisualizer->SetOutputImage(resized_test, 0);
		}
		
	}
	
	bool Tracker::TrackWithPrevFrame(Frame* prev, Frame* cur, float thMaxDesc, float thMinDesc){
		cur->reset_map_points();
		int res =SearchPoints::SearchFrameByProjection(prev, cur, thMaxDesc, thMinDesc);

		if (res < 20) {
			cur->reset_map_points();
			res = SearchPoints::SearchFrameByProjection(prev, cur, thMaxDesc, thMinDesc, 30.0);
		}
		if (res < 20){
			std::cout << "Matching prev fail!!!" << std::endl;
			return false;
		}
		int nopt = Optimizer::PoseOptimization(cur);
		
		// Discard outliers
		int nmatchesMap = 0;
		for (int i = 0; i<cur->N; i++)
		{
			if (cur->mvpMapPoints[i])
			{
				if (cur->mvbOutliers[i])
				{
					MapPoint* pMP = cur->mvpMapPoints[i];

					cur->mvpMapPoints[i] = nullptr;
					cur->mvbOutliers[i] = false;
					pMP->mbTrackInView = false;
					pMP->mnLastFrameSeen = cur->mnFrameID;
					res--;
				}
				else if (cur->mvpMapPoints[i]->Observations()>0)
					nmatchesMap++;
			}
		}
		if (nmatchesMap < 10)
			std::cout << "?????????????" << std::endl;
		return nmatchesMap>=10;
	}
	bool Tracker::TrackWithKeyFrame(KeyFrame* ref, Frame* cur){
		return false;
	}
	int Tracker::TrackWithLocalMap(User* user, Frame* cur, float thMaxDesc, float thMinDesc){
		LocalMap* pLocalMap = new LocalCovisibilityMap();
		std::vector<MapPoint*> vpLocalMPs;
		std::vector<KeyFrame*> vpLocalKFs;
		//std::cout << "Track::LocalMap::Update::start" << std::endl;
		pLocalMap->UpdateLocalMap(user, cur, vpLocalKFs, vpLocalMPs);
		//std::cout << "Track::LocalMap::Update::end" << std::endl;
		//update visible

		int nMatch = UpdateVisiblePoints(cur, vpLocalMPs);
		//std::cout << "Track::LocalMap::Update::Visible::end" << std::endl;
		if (nMatch == 0)
			return 0;

		float thRadius = 1.0;
		if (cur->mnFrameID < user->mnLastRelocFrameId + 2)
			thRadius = 5.0;

		int a = SearchPoints::SearchMapByProjection(cur, vpLocalMPs, thMaxDesc, thMinDesc, thRadius);
		std::cout << "match local map = " << vpLocalKFs.size() << " " << vpLocalMPs.size() <<", "<<nMatch<< "=" << a << std::endl;
		Optimizer::PoseOptimization(cur);
		return UpdateFoundPoints(cur);
	}

	bool Tracker::Relocalization(Map* map, User* user, Frame* cur, float thMinDesc) {
		std::cout << "Relocalization!!!" << std::endl;

		cur->ComputeBoW();

		std::vector<KeyFrame*> vpCandidateKFs = map->mpKeyFrameDB->DetectRelocalizationCandidates(cur);
		std::cout << "Candidates = " << vpCandidateKFs.size() << std::endl;
		if (vpCandidateKFs.empty())
			return false;

		const int nKFs = vpCandidateKFs.size();

		// We perform first an ORB matching with each candidate
		// If enough matches are found we setup a PnP solver

		std::vector<PnPSolver*> vpPnPsolvers;
		vpPnPsolvers.resize(nKFs);

		std::vector<std::vector<MapPoint*> > vvpMapPointMatches;
		vvpMapPointMatches.resize(nKFs);

		std::vector<bool> vbDiscarded;
		vbDiscarded.resize(nKFs);

		int nCandidates = 0;

		for (int i = 0; i<nKFs; i++)
		{
			KeyFrame* pKF = vpCandidateKFs[i];
			if (pKF->isBad())
				vbDiscarded[i] = true;
			else
			{
				
				int nmatches = SearchPoints::SearchFrameByBoW(cur->matcher, pKF, cur, vvpMapPointMatches[i], thMinDesc, 0.75);
				std::cout << "BoW Match = " << nmatches << std::endl;
				if (nmatches<15)
				{
					vbDiscarded[i] = true;
					continue;
				}
				else
				{
					PnPSolver* pSolver = new PnPSolver(cur, vvpMapPointMatches[i]);
					pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
					vpPnPsolvers[i] = pSolver;
					nCandidates++;
				}
			}
		}
		std::cout << "SolvePnP::Start" << std::endl;
		// Alternatively perform some iterations of P4P RANSAC
		// Until we found a camera pose supported by enough inliers
		bool bMatch = false;

		while (nCandidates>0 && !bMatch)
		{
			for (int i = 0; i<nKFs; i++)
			{
				if (vbDiscarded[i])
					continue;

				// Perform 5 Ransac Iterations
				std::vector<bool> vbInliers;
				int nInliers;
				bool bNoMore;
				std::cout << "PnP::Iterate::Start" << std::endl;
				PnPSolver* pSolver = vpPnPsolvers[i];
				cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);
				std::cout << "PnP::Iterate::Start" << std::endl;
				// If Ransac reachs max. iterations discard keyframe
				if (bNoMore)
				{
					vbDiscarded[i] = true;
					nCandidates--;
				}

				// If a Camera Pose is computed, optimize
				if (!Tcw.empty())
				{
					cur->SetPose(Tcw);

					std::set<MapPoint*> sFound;

					const int np = vbInliers.size();

					for (int j = 0; j<np; j++)
					{
						if (vbInliers[j])
						{
							cur->mvpMapPoints[j] = vvpMapPointMatches[i][j];
							sFound.insert(vvpMapPointMatches[i][j]);
						}
						else
							cur->mvpMapPoints[j] = nullptr;
					}

					int nGood = Optimizer::PoseOptimization(cur);

					if (nGood<10)
						continue;

					for (int io = 0; io<cur->N; io++)
						if (cur->mvbOutliers[io])
							cur->mvpMapPoints[io] = static_cast<MapPoint*>(NULL);

					// If few inliers, search by projection in a coarse window and optimize again
					if (nGood<50)
					{
						int nadditional = SearchPoints::SearchFrameByProjection(cur->matcher,cur, vpCandidateKFs[i], sFound, 10, 100);
						if (nadditional + nGood >= 50)
						{
							nGood = Optimizer::PoseOptimization(cur);

							// If many inliers but still not enough, search by projection again in a narrower window
							// the camera has been already optimized with many points
							if (nGood>30 && nGood<50)
							{
								sFound.clear();
								for (int ip = 0; ip<cur->N; ip++)
									if (cur->mvpMapPoints[ip])
										sFound.insert(cur->mvpMapPoints[ip]);
								nadditional = SearchPoints::SearchFrameByProjection(cur->matcher,cur, vpCandidateKFs[i], sFound, 3, 64);
								// Final optimization
								if (nGood + nadditional >= 50)
								{
									nGood = Optimizer::PoseOptimization(cur);

									for (int io = 0; io<cur->N; io++)
										if (cur->mvbOutliers[io])
											cur->mvpMapPoints[io] = NULL;
								}
							}
						}
					}


					// If the pose is supported by enough inliers stop ransacs and continue
					if (nGood >= 50)
					{
						bMatch = true;
						break;
					}
				}
			}
		}
		std::cout << "SolvePnP::End" << std::endl;
		if (!bMatch)
		{
			return false;
		}
		else
		{
			user->mnLastRelocFrameId = cur->mnFrameID;
			return true;
		}

		return false;
	}

	int Tracker::UpdateVisiblePoints(Frame* cur, std::vector<MapPoint*> vpLocalMPs) {
		// Do not search map points already matched
		int nFrameID = cur->mnFrameID;
		for (auto vit = cur->mvpMapPoints.begin(), vend = cur->mvpMapPoints.end(); vit != vend; vit++)
		{
			MapPoint* pMP = *vit;
			if (pMP)
			{
				if (pMP->isBad())
				{
					*vit = static_cast<MapPoint*>(nullptr);
				}
				else
				{
					pMP->IncreaseVisible();
					pMP->mnLastFrameSeen = nFrameID;
					pMP->mbTrackInView = false;
				}
			}
		}

		int nToMatch = 0;

		// Project points in frame and check its visibility
		for (auto vit = vpLocalMPs.begin(), vend = vpLocalMPs.end(); vit != vend; vit++)
		{
			MapPoint* pMP = *vit;
			if (pMP->mnLastFrameSeen == nFrameID || pMP->isBad())
				continue;
			// Project (this fills MapPoint variables for matching)
			if (cur->is_in_frustum(pMP, 0.5))
			{
				pMP->IncreaseVisible();
				nToMatch++;
			}
		}
		return nToMatch;
	}

	int Tracker::UpdateFoundPoints(Frame* cur, bool bOnlyTracking) {
		int nres = 0;
		// Update MapPoints Statistics
		for (int i = 0; i<cur->N; i++)
		{
			if (cur->mvpMapPoints[i])
			{
				if (!cur->mvbOutliers[i])
				{
					cur->mvpMapPoints[i]->IncreaseFound();
					if (!bOnlyTracking)
					{
						if (cur->mvpMapPoints[i]->Observations()>0)
							nres++;
					}
					else
						nres++;
				}
			}
		}
		return nres;
	}

	bool Tracker::NeedNewKeyFrame(Map* map, LocalMapper* mapper, Frame* cur, KeyFrame* ref, int nMatchesInliers, int nLastKeyFrameId, int nLastRelocFrameID, bool bOnlyTracking, int nMaxFrames, int nMinFrames)
	{
		if (bOnlyTracking)
			return false;

		// If Local Mapping is freezed by a Loop Closure do not insert keyframes
		if (mapper->isStopped() || mapper->stopRequested())
			return false;

		const int nKFs = map->GetNumKeyFrames();

		// Do not insert keyframes if not enough frames have passed from last relocalisation
		if (cur->mnFrameID<nLastRelocFrameID + nMaxFrames && nKFs>nMaxFrames)
			return false;

		// Tracked MapPoints in the reference keyframe
		int nMinObs = 3;
		if (nKFs <= 2)
			nMinObs = 2;
		int nRefMatches = ref->TrackedMapPoints(nMinObs);

		// Local Mapping accept keyframes?
		bool bLocalMappingIdle = map->mnNumMappingFrames == 0;
		// Thresholds
		float thRefRatio = 0.9f;
		
		// Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
		const bool c1a = cur->mnFrameID >= nLastKeyFrameId + nMaxFrames;
		// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
		const bool c1b = cur->mnFrameID >= nLastKeyFrameId + nMinFrames && bLocalMappingIdle;
		//Condition 1c: tracking is weak
		//const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose);
		// Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
		const bool c2 = (nMatchesInliers<nRefMatches*thRefRatio) && nMatchesInliers>15;

		if ((c1a || c1b) && c2)
		{
			// If the mapping accepts keyframes, insert keyframe.
			// Otherwise send a signal to interrupt BA
			if (bLocalMappingIdle)
			{
				return true;
			}
			map->InterruptBA();
			return false;
		}
		return false;
	}

	void Tracker::CreateNewKeyFrame(ThreadPool::ThreadPool* pool, SLAM* system, Map* map, LocalMapper* mapper, Frame* cur, User* user)
	{
		if (!mapper->SetNotStop(true))
			return;
		KeyFrame* pKF = new KeyFrame(cur, map);
		user->mnReferenceKeyFrameID = pKF->mnId;
		user->mnLastKeyFrameID = cur->mnFrameID;
		user->mapKeyFrames[pKF->mnId] = pKF;
		pool->EnqueueJob(LocalMapper::ProcessMapping, pool, system, map, pKF);
		mapper->SetNotStop(false);
	}
}