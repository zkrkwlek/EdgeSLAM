#include <Tracker.h>
#include <SLAM.h>
#include <Initializer.h>
#include <LocalMapper.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <ObjectFrame.h>
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
#include <Segmentator.h>
#include <WebAPI.h>
#include <Converter.h>

#include <chrono>
namespace EdgeSLAM {
	Tracker::Tracker(){}
	Tracker::~Tracker(){}

	cv::Point2f CalcLinePoint(float val, cv::Mat mLine, bool opt) {
		float x, y;
		if (opt) {
			x = 0.0;
			y = val;
			if (mLine.at<float>(0) != 0)
				x = (-mLine.at<float>(2) - mLine.at<float>(1)*y) / mLine.at<float>(0);
		}
		else {
			y = 0.0;
			x = val;
			if (mLine.at<float>(1) != 0)
				y = (-mLine.at<float>(2) - mLine.at<float>(0)*x) / mLine.at<float>(1);
		}

		return cv::Point2f(x, y);
	}

	void Tracker::DownloadKeyPoints(SLAM* system, User* user, int id) {
		std::stringstream ss;
		ss << "/Load?keyword=Keypoints" << "&id=" << id << "&src=" << user->userName;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();

		cv::Mat fdata = cv::Mat::zeros(n2/8, 2, CV_32FC1);
		std::memcpy(fdata.data, res.data(), res.size());
		user->mapKeyPoints.Update(id, fdata.clone());

		delete mpAPI;
	}

	void Tracker::UpdateDeviceGyro(SLAM* system, User* user, int id) {
		std::stringstream ss;
		ss << "/Load?keyword=Gyro" << "&id=" << id << "&src=" << user->userName;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();

		cv::Mat fdata = cv::Mat::zeros(3, 3, CV_32FC1);
		std::memcpy(fdata.data, res.data(), res.size());
		user->UpdateGyro(fdata);

		delete mpAPI;
	}
	void Tracker::Track(ThreadPool::ThreadPool* pool, SLAM* system, int id, User* user, double ts) {
		if (user->mbProgress)
			return;
		user->mbProgress = true;
		auto cam = user->mpCamera;
		auto map = user->mpMap;

		//time analysis
		float t_local = 1000.0;
		float t_send = 1000.0;
		float t_prev = 1000.0;
		float t_init = 1000.0;
		float t_frame = 1000.0;
		LocalMap* pLocalMap = new LocalCovisibilityMap();

		//std::cout << "Frame = " << user->userName <<" "<<id<< "=start!!" << std::endl;
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		////receive image
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		std::stringstream ss;
		ss << "/Load?keyword=Image" << "&id=" << id << "&src=" << user->userName;
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();
		cv::Mat temp = cv::Mat::zeros(n2, 1, CV_8UC1);
		std::memcpy(temp.data, res.data(), res.size());
		cv::Mat img = cv::imdecode(temp, cv::IMREAD_COLOR);
		
		if (img.empty())
		{
			std::cout << "Decoding image = " << id << "=" << img.size() << " " << img.type() << " " << n2 << " = " << (int)temp.at<uchar>(n2 - 1) << " " << (int)temp.at<uchar>(n2 - 2) << std::endl;
			std::cout << "Error = Decoding image = " << id << std::endl;
			user->mbProgress = false;
			return;
		}
		
		std::chrono::high_resolution_clock::time_point received = std::chrono::high_resolution_clock::now();
		////receive image
		/////save image
		/*std::stringstream sss;
		sss << "../bin/img/" << user->userName << "/" << id << "_color.jpg";
		cv::imwrite(sss.str(), img);*/
		/////save image
		
		/////KP 여기도 플래그 추가하기
		Frame* frame = nullptr;
		std::chrono::high_resolution_clock::time_point t_frame_s = std::chrono::high_resolution_clock::now();
		//if (user->mbMapping) {
		//	while (user->mapKeyPoints.Count(id) == 0)
		//	{
		//		continue;
		//	}
		//	//KP
		//	cv::Mat data = user->mapKeyPoints.Get(id);
		//	user->mapKeyPoints.Erase(id);
		//	frame = new Frame(img, data, cam, id, ts);
		//}
		//else {
		//	frame = new Frame(img, cam, id, ts);
		//}
		frame = new Frame(img, cam, id, ts);
		std::chrono::high_resolution_clock::time_point t_frame_e = std::chrono::high_resolution_clock::now();
		auto du_frame = std::chrono::duration_cast<std::chrono::milliseconds>(t_frame_e - t_frame_s).count();
		t_frame = du_frame / 1000.0;
		
		//user->mapFrames.Update(frame->mnFrameID, frame);
		
		
		//std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);

		auto mapState = map->GetState();
		auto userState = user->GetState();
		auto trackState = UserState::NotEstimated;
		int nInliers = 0;
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
				map->mvpKeyFrameOrigins.push_back(kf1);
				/*user->mapKeyFrames[kf1->mnId] = kf1;
				user->mapKeyFrames[kf2->mnId] = kf2;*/
				user->mpRefKF = kf2;
				user->mnLastKeyFrameID = frame->mnFrameID;
				pool->EnqueueJob(LocalMapper::ProcessMapping, pool, system, map, kf1);
				pool->EnqueueJob(LocalMapper::ProcessMapping, pool, system, map, kf2);

				for (int i = 0; i < frame->N; i++)
				{
					if (frame->mvpMapPoints[i])
					{
						if (!frame->mvbOutliers[i])
						{
							nInliers++;
						}
					}
				}
			}
		}else if (mapState == MapState::Initialized) {
			bool bTrack = false;
			if (userState == UserState::NotEstimated || userState == UserState::Failed) {
				//global localization
				//set reference keyframe and last keyframe
				frame->reset_map_points();
				nInliers = Tracker::Relocalization(map, user, frame, system->mpFeatureTracker->min_descriptor_distance);
				if (nInliers >= 50)
				{
					bTrack = true;
					user->mnLastRelocFrameId = frame->mnFrameID;
					//trackState = UserState::Success;
				}
				std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
				auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				float t_test1 = du_test1 / 1000.0;
				
				int N = system->GetConnectedDevice();
				system->ProcessingTime.Get("reloc")[N]->add(t_test1);
			}
			else {
				if (userState == UserState::Success) {
					
					//std::cout << "Tracker::Start" << std::endl;
					//auto f_ref = user->mapFrames[user->mnPrevFrameID];
					auto prevFrame = user->prevFrame;
					prevFrame->check_replaced_map_points();

					cv::Mat Tpredict;
					if (user->mbIMU) {
						auto Rgyro = user->GetGyro();
						cv::Mat Tgyro = cv::Mat::eye(4, 4, CV_32FC1);
						Rgyro.copyTo(Tgyro.rowRange(0, 3).colRange(0, 3));
						Tpredict = Tgyro * user->GetPose();
					}
					else {
						Tpredict = user->PredictPose();
					}
					frame->SetPose(Tpredict);
					std::chrono::high_resolution_clock::time_point t_prev_start = std::chrono::high_resolution_clock::now();
					bTrack = Tracker::TrackWithPrevFrame(prevFrame, frame, system->mpFeatureTracker->max_descriptor_distance, system->mpFeatureTracker->min_descriptor_distance);
					std::chrono::high_resolution_clock::time_point t_prev_end = std::chrono::high_resolution_clock::now();
					auto du_prev = std::chrono::duration_cast<std::chrono::milliseconds>(t_prev_end - t_prev_start).count();
					t_prev = du_prev / 1000.0;

					auto du_init = std::chrono::duration_cast<std::chrono::milliseconds>(t_prev_start - received).count();
					t_init = du_init / 1000.0;
					if (!bTrack) {
						std::cout << "track with reference frame :: start" << std::endl;
						std::cout << "track with reference frame :: end" << std::endl;
					}
					else {
						
					}
				}
				/*if (userState == UserState::Failed) {
				frame->reset_map_points();
				nInliers = system->mpTracker->Relocalization(map, user, frame, system->mpFeatureTracker->min_descriptor_distance);
				if (nInliers >= 50) {
				bTrack = true;
				user->mnLastRelocFrameId = frame->mnFrameID;
				}
				}*/
			}
			
			if (bTrack) {
				std::chrono::high_resolution_clock::time_point t_local_start = std::chrono::high_resolution_clock::now();
				nInliers = Tracker::TrackWithLocalMap(pLocalMap, user, frame, system->mpFeatureTracker->max_descriptor_distance, system->mpFeatureTracker->min_descriptor_distance);
				std::chrono::high_resolution_clock::time_point t_local_end = std::chrono::high_resolution_clock::now();
				auto du_local = std::chrono::duration_cast<std::chrono::milliseconds>(t_local_end - t_local_start).count();
				t_local = du_local / 1000.0;
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
			if (!bTrack) {
				trackState = UserState::Failed;
				nInliers = 0;
				cv::Mat Rt = cv::Mat::eye(4, 4, CV_32FC1);
				frame->SetPose(Rt);
			}
			if (bTrack)
				trackState = UserState::Success;
			
			pool->EnqueueJob(Tracker::SendDeviceTrackingData, system, user, pLocalMap, frame, nInliers, id, ts);
			
		}
		
		user->SetState(trackState);
		if (trackState == UserState::Success) {
			
			////오브젝트 검출 요청
			//Segmentator::RequestObjectDetection(user->userName, frame->mnFrameID);
			
			//pose update
			cv::Mat T = frame->GetPose();
			user->UpdatePose(T, ts);
			//check keyframe
			if (user->mbMapping && user->mpRefKF) {
				if (Tracker::NeedNewKeyFrame(map, system->mpLocalMapper, frame, user->mpRefKF, nInliers, user->mnLastKeyFrameID.load(), user->mnLastRelocFrameId.load())) {
					Tracker::CreateNewKeyFrame(pool, system, map, system->mpLocalMapper, frame, user);
					Segmentator::RequestSegmentation(user->userName, frame->mnFrameID);
				}
			}
			
			////frame line visualization
			/*if (!user->mbMapping) {
				cv::Mat R, t;
				R = frame->GetRotation();
				t = frame->GetTranslation();
				float m1,m2;
				cv::Mat line1, line2;
				line1 = Segmentator::LineProjection(R, t, Segmentator::Lw1, frame->mpCamera->Kfluker, m1);
				m1 = -line1.at<float>(0) / line1.at<float>(1);
				bool bSlopeOpt1 = abs(m1) > 1.0;
				float val1;
				if (bSlopeOpt1)
					val1 = 360.0;
				else
					val1 = 640.0;
				auto sPt1 = CalcLinePoint(0.0, line1, bSlopeOpt1);
				auto ePt1 = CalcLinePoint(val1, line1, bSlopeOpt1);
				cv::line(img, sPt1, ePt1, cv::Scalar(255, 0, 255),3);

				line2 = Segmentator::LineProjection(R, t, Segmentator::Lw2, frame->mpCamera->Kfluker, m2);
				m2 = -line2.at<float>(0) / line2.at<float>(1);
				bool bSlopeOpt2 = abs(m2) > 1.0;
				float val2;
				if (bSlopeOpt2)
					val2 = 360.0;
				else
					val2 = 640.0;
				auto sPt2 = CalcLinePoint(0.0, line2, bSlopeOpt2);
				auto ePt2 = CalcLinePoint(val2, line2, bSlopeOpt2);
				cv::line(img, sPt2, ePt2, cv::Scalar(255, 255, 0),3);
			}*/
			////frame line visualization
		}
		
		//int tempID = user->mnPrevFrameID;
		user->mnPrevFrameID = user->mnCurrFrameID.load();
		user->mnCurrFrameID = frame->mnFrameID;

		if (mapState == MapState::Initialized && user->prevFrame)
			delete user->prevFrame;
		user->prevFrame = frame;
		user->mbProgress = false; 
		
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(received - start).count();
		float t_test1 = du_test1 / 1000.0;

		auto du_test2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - received).count();
		float t_test2 = du_test2 / 1000.0;

		delete pLocalMap;
		delete mpAPI;

		int N = system->GetConnectedDevice();
		system->ProcessingTime.Get("download")[N]->add(t_test1);
		system->ProcessingTime.Get("tracking")[N]->add(t_test2);
		if (mapState == MapState::Initialized && !user->mbMapping) {
			int ntemp = userState == UserState::Success ? 1 : 0;
			system->SuccessRatio.Get("skipframe")[user->mnSkip]->increase(ntemp);

			if (user->mbAsyncTest) {
				system->SuccessRatio.Get("async")[user->mnQuality]->increase(ntemp);
			}

		}
		std::cout << "Frame = " << user->userName << " : " << id << ", Matches = " << nInliers << ", time =" << t_test1 <<", "<< t_test2<<"="<< t_local <<" "<<t_prev<<" "<< t_init <<" "<<t_frame<< std::endl;
		
		//system->UpdateTrackingTime(t_test1);

		////visualization
		if (mapState == MapState::Initialized  && user->GetVisID() <= 3 && userState != UserState::NotEstimated) {

			cv::Scalar color = Segmentator::mvObjectLabelColors[user->GetVisID()+1];

			for (int i = 0; i < frame->mvKeys.size(); i++) {
				auto pMP = frame->mvpMapPoints[i];
				//cv::Scalar color = cv::Scalar(255, 0, 255);
				int r = 2;
				if (pMP && !pMP->isBad())
				{

					/*if (Segmentator::ObjectPoints.Count(pMP->mnId))
					{
						int label = Segmentator::ObjectPoints.Get(pMP->mnId)->GetLabel();
						color = Segmentator::mvObjectLabelColors[label];
					}
					else {
						color.val[1] = 255;
						color.val[2] = 0;
						
					}
					r++;*/
					/*color.val[1] = 255;
					color.val[2] = 0;*/
					cv::circle(img, frame->mvKeys[i].pt, r, color, -1);
				}
			}
			/*if (user->objFrames.Count(tempID)) {
				ObjectFrame* objFrame = user->objFrames.Get(tempID);
				for (auto iter = objFrame->mapObjects.begin(), iend = objFrame->mapObjects.end(); iter != iend; iter++) {
					auto box = iter->second;
					cv::rectangle(img, box->rect, cv::Scalar(0, 255, 255));
				}
			}
			if (user->objFrames.Count(id)) {
				ObjectFrame* objFrame = user->objFrames.Get(id);
				for (auto iter = objFrame->mapObjects.begin(), iend = objFrame->mapObjects.end(); iter != iend; iter++) {
					auto box = iter->second;
					cv::rectangle(img, box->rect, cv::Scalar(255, 0, 255));
				}
			}
			else {
				std::cout << std::endl << std::endl << std::endl << std::endl;
			}*/
			

			system->mpVisualizer->ResizeImage(img, img);
			system->mpVisualizer->SetOutputImage(img, user->GetVisID());

			/////save image
			/*std::stringstream sss;
			sss << "../../bin/img/" << user->userName << "/Track/" << id << ".jpg";
			cv::imwrite(sss.str(), img);*/
			/////save image
		}
		////visualization
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
		//int nmatchesMap = DiscardOutliers(cur);
		return nopt >=10;
	}
	bool Tracker::TrackWithKeyFrame(KeyFrame* ref, Frame* cur){
		return false;
	}
	int Tracker::TrackWithLocalMap(LocalMap* pLocalMap, User* user, Frame* cur, float thMaxDesc, float thMinDesc){
		
		pLocalMap->UpdateLocalMap(user, cur);
		
		//update visible
		int nMatch = Tracker::UpdateVisiblePoints(cur, pLocalMap->mvpLocalMPs, pLocalMap->mvpLocalTPs);
		if (nMatch == 0)
			return 0;

		float thRadius = 1.0;
		if (cur->mnFrameID < user->mnLastRelocFrameId + 2)
			thRadius = 5.0;

		int a = SearchPoints::SearchMapByProjection(cur, pLocalMap->mvpLocalMPs, pLocalMap->mvpLocalTPs, thMaxDesc, thMinDesc, thRadius);
		Optimizer::PoseOptimization(cur);
		return Tracker::UpdateFoundPoints(cur);
	}

	int Tracker::Relocalization(Map* map, User* user, Frame* cur, float thMinDesc) {

		cur->ComputeBoW();

		std::vector<KeyFrame*> vpCandidateKFs = map->mpKeyFrameDB->DetectRelocalizationCandidates(cur);
		
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

		//std::vector<int> vnGoods(nKFs,0);

		int nCandidates = 0;

		for (int i = 0; i<nKFs; i++)
		{
			KeyFrame* pKF = vpCandidateKFs[i];
			if (pKF->isBad())
				vbDiscarded[i] = true;
			else
			{
				
				int nmatches = SearchPoints::SearchFrameByBoW(pKF, cur, vvpMapPointMatches[i], thMinDesc, 0.75);
				if (nmatches<15)
				{
					vbDiscarded[i] = true;
					continue;
				}
				else {
					nCandidates++;
				}
				/*else
				{
					PnPSolver* pSolver = new PnPSolver(cur, vvpMapPointMatches[i]);
					pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
					vpPnPsolvers[i] = pSolver;
					nCandidates++;
				}*/
			}
		}
		//std::cout << "Reloc = temp = "<< nKFs<<" "<<nCandidates << std::endl;
		// Alternatively perform some iterations of P4P RANSAC
		// Until we found a camera pose supported by enough inliers
		bool bMatch = false;
		int nGood = 0;
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
				
				if (vvpMapPointMatches[i].size() < 10) {
					vbDiscarded[i] = true;
					nCandidates--;
					continue;
				}

				cur->SetPose(vpCandidateKFs[i]->GetPose());
				std::set<MapPoint*> sFound;
				for (size_t j = 0, jend = vvpMapPointMatches[i].size(); j <jend; j++)
				{
					MapPoint* pMP = vvpMapPointMatches[i][j];

					if (pMP && !pMP->isBad())
					{
						cur->mvpMapPoints[j] = vvpMapPointMatches[i][j];
						sFound.insert(vvpMapPointMatches[i][j]);
					}
				}
				nGood = Optimizer::PoseOptimization(cur);
				//vnGoods[i] = nGood;
				//std::cout << "relocalization="<<i<<"=init::ngood=" << nGood << std::endl;
				if (nGood < 10){
					vbDiscarded[i] = true;
					nCandidates--;
					continue;
				}
				for (int io = 0; io<cur->N; io++)
					if (cur->mvbOutliers[io])
						cur->mvpMapPoints[io] = static_cast<MapPoint*>(NULL);

				if (nGood<50)
				{
					int nadditional = SearchPoints::SearchFrameByProjection(cur, vpCandidateKFs[i], sFound, 10, 100);
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
							nadditional = SearchPoints::SearchFrameByProjection(cur, vpCandidateKFs[i], sFound, 3, 64);
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
				std::cout << "relocalization=" << i << "=final::ngood=" << nGood << std::endl;
				/*if (vnGoods[i] == nGood) {
					vbDiscarded[i] = true;
					nCandidates--;
					continue;
				}*/
				//vnGoods[i] = nGood;
				if (nGood >= 50)
				{
					bMatch = true;
					break;
				}
				else {
					vbDiscarded[i] = true;
					nCandidates--;
				}
				
				continue;
			}
		}
		//std::cout << "End::relocalization!!" << std::endl;
		return nGood;
		/*if (!bMatch)
		{
			return false;
		}
		else
		{
			user->mnLastRelocFrameId = cur->mnFrameID;
			return true;
		}*/
	}
	
	int Tracker::UpdateVisiblePoints(Frame* cur, std::vector<MapPoint*> vpLocalMPs, std::vector<TrackPoint*> vpLocalTPs) {
		// Do not search map points already matched
		for (int i = 0; i<cur->N; i++)
		{
			if (cur->mvpMapPoints[i])
			{
				MapPoint* pMP = cur->mvpMapPoints[i];
				if (!pMP || pMP->isBad() || cur->mvbOutliers[i]) {
					cur->mvpMapPoints[i] = nullptr;
					cur->mvbOutliers[i] = false;
				}
				else {
					pMP->IncreaseVisible();
				}
				cur->mspMapPoints.insert(pMP);
			}
		}

		int nToMatch = 0;

		// Project points in frame and check its visibility
		for(size_t i = 0, iend = vpLocalMPs.size(); i < iend; i++)
		//for (auto vit = vpLocalMPs.begin(), vend = vpLocalMPs.end(); vit != vend; vit++)
		{
			MapPoint* pMP = vpLocalMPs[i];
			TrackPoint* pTP = vpLocalTPs[i];
			if (cur->mspMapPoints.count(pMP) || pMP->isBad())
				continue;
			// Project (this fills MapPoint variables for matching)
			if (cur->is_in_frustum(pMP, pTP, 0.5))
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
		if (map->isStopped() || map->stopRequested())
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
		
		if (!map->SetNotStop(true))
			return;
		KeyFrame* pKF = new KeyFrame(cur, map);
		user->mpRefKF = pKF;
		user->mnLastKeyFrameID = cur->mnFrameID;
		pool->EnqueueJob(LocalMapper::ProcessMapping, pool, system, map, pKF);
		map->SetNotStop(false);
		
		user->KeyFrames.Update(cur->mnFrameID, pKF);
	}

	void Tracker::SendTrackingResults(SLAM* system, User* user, int nFrameID, int n, cv::Mat R, cv::Mat t) {
		
		auto q = Converter::toQuaternion(R.t());
		cv::Mat data = cv::Mat::zeros(400, 1, CV_32FC1);
		data.at<float>(0) = (float)n;
		data.at<float>(1) = q[0];
		data.at<float>(2) = q[1];
		data.at<float>(3) = q[2];
		data.at<float>(4) = q[3];
		data.at<float>(5) = t.at<float>(0);
		data.at<float>(6) = t.at<float>(1);
		data.at<float>(7) = t.at<float>(2);
		for (int i = 8; i < data.rows; i++) {
			data.at<float>(i) = i;
		}
		
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		std::stringstream ss;
		ss << "/Store?keyword=MappingResult&id=" << nFrameID << "&src=" << user->userName;// << "&type2=" << user->userName;
		auto res = mpAPI->Send(ss.str(), data.data, sizeof(float)*data.rows);

		delete mpAPI;
	}

	void Tracker::SendDeviceTrackingData(SLAM* system, User* user, LocalMap* pLocalMap, Frame* frame, int nInlier, int id, double ts) {
		
		////data
		cv::Mat T = frame->GetPose();
		cv::Mat data = cv::Mat::zeros(13, 1, CV_32FC1); //inlier, pose + point2f, octave, angle, point3f
		
		
		int nDataIdx = 1;
		data.at<float>(nDataIdx++) = T.at<float>(0, 0);
		data.at<float>(nDataIdx++) = T.at<float>(0, 1);
		data.at<float>(nDataIdx++) = T.at<float>(0, 2);
		data.at<float>(nDataIdx++) = T.at<float>(1, 0);
		data.at<float>(nDataIdx++) = T.at<float>(1, 1);
		data.at<float>(nDataIdx++) = T.at<float>(1, 2);
		data.at<float>(nDataIdx++) = T.at<float>(2, 0);
		data.at<float>(nDataIdx++) = T.at<float>(2, 1);
		data.at<float>(nDataIdx++) = T.at<float>(2, 2);
		data.at<float>(nDataIdx++) = T.at<float>(0, 3);
		data.at<float>(nDataIdx++) = T.at<float>(1, 3);
		data.at<float>(nDataIdx++) = T.at<float>(2, 3);

		if (nInlier > 0){
			int nres = 0;
			for (int i = 0; i < frame->N; i++)
			{
				if (frame->mvpMapPoints[i])
				{
					if (!frame->mvbOutliers[i] && !frame->mvpMapPoints[i]->isBad())
					{
						int nDataIdx = 0;
						auto kp = frame->mvKeys[i];
						auto mp = frame->mvpMapPoints[i]->GetWorldPos();
						int octave = kp.octave;
						cv::Mat temp = cv::Mat::zeros(8, 1, CV_32FC1);
						temp.at<float>(nDataIdx++) = kp.pt.x;
						temp.at<float>(nDataIdx++) = kp.pt.y;
						temp.at<float>(nDataIdx++) = (float)kp.octave;
						temp.at<float>(nDataIdx++) = kp.angle;
						temp.at<float>(nDataIdx++) = (float)frame->mvpMapPoints[i]->mnId;
						temp.at<float>(nDataIdx++) = mp.at<float>(0);
						temp.at<float>(nDataIdx++) = mp.at<float>(1);
						temp.at<float>(nDataIdx++) = mp.at<float>(2);
						data.push_back(temp);
						nres++;
					}
				}
			}
			data.at<float>(0) = (float)nres;
		}
		else{
			for (int i = 0; i < 500; i++) {
				cv::Mat temp = cv::Mat::ones(8, 1, CV_32FC1);
				data.push_back(temp);
			}
			data.at<float>(0) = 0.0;
		}
		{
			WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
			std::stringstream ss;
			ss << "/Store?keyword=ReferenceFrame&id=" << id << "&src=" << user->userName <<"&ts="<<std::fixed<< std::setprecision(6) <<ts<< "&type2=" << user->userName;
			std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
			auto res = mpAPI->Send(ss.str(), data.data, data.rows * sizeof(float));
			std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();
			delete mpAPI;
			
			auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
			float t_test1 = du_test1 / 1000.0;
			int N = system->GetConnectedDevice();
			system->ProcessingTime.Get("upload")[N]->add(t_test1);
		}
	
		////data
		
	}
}