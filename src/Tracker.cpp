#include <Tracker.h>
#include <SLAM.h>
#include <Initializer.h>
#include <LocalMapper.h>
#include <LoopCloser.h>
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
#include <Utils.h>
#include <KalmanFilter.h>

#include <chrono>
namespace EdgeSLAM{
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
	void Tracker::ProcessDevicePosition(SLAM* system, std::string user, int id, double ts) {
		auto pUser = system->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;
		
		std::stringstream ss;
		ss << "/Load?keyword=DevicePosition" << "&id=" << id << "&src=" << user;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();

		cv::Mat fdata = cv::Mat::zeros(4, 3, CV_32FC1);
		std::memcpy(fdata.data, res.data(), res.size());
		pUser->mvDeviceTrajectories.push_back(fdata.clone());
		pUser->mvDeviceTimeStamps.push_back(ts);
		fdata.release();
		pUser->mnUsed--;
	}

	void Tracker::UpdateDeviceGyro(SLAM* system, std::string user, int id) {
		auto pUser = system->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;
		std::stringstream ss;
		ss << "/Load?keyword=Gyro" << "&id=" << id << "&src=" << user;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();

		cv::Mat fdata = cv::Mat::zeros(3, 3, CV_32FC1);
		std::memcpy(fdata.data, res.data(), res.size());
		pUser->UpdateGyro(fdata);
		pUser->mnUsed--;
		delete mpAPI;
	}

	void Tracker::CreatePointsOXR(KeyFrame* pRefKeyframe, KeyFrame* pCurKeyframe, Frame* pCurFrame, Map* pMap) {
		
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		const float& fx1 = pRefKeyframe->fx;
		const float& fy1 = pRefKeyframe->fy;
		const float& cx1 = pRefKeyframe->cx;
		const float& cy1 = pRefKeyframe->cy;
		const float& invfx1 = pRefKeyframe->invfx;
		const float& invfy1 = pRefKeyframe->invfy;

		const float& fx2 = pCurKeyframe->fx;
		const float& fy2 = pCurKeyframe->fy;
		const float& cx2 = pCurKeyframe->cx;
		const float& cy2 = pCurKeyframe->cy;
		const float& invfx2 = pCurKeyframe->invfx;
		const float& invfy2 = pCurKeyframe->invfy;

		cv::Mat Rcw1 = pRefKeyframe->GetRotation();
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat tcw1 = pRefKeyframe->GetTranslation();
		cv::Mat Tcw1(3, 4, CV_32F);
		Rcw1.copyTo(Tcw1.colRange(0, 3));
		tcw1.copyTo(Tcw1.col(3));

		cv::Mat Rcw2 = pCurKeyframe->GetRotation();
		cv::Mat Rwc2 = Rcw2.t();
		cv::Mat tcw2 = pCurKeyframe->GetTranslation();
		cv::Mat Tcw2(3, 4, CV_32F);
		Rcw2.copyTo(Tcw2.colRange(0, 3));
		tcw2.copyTo(Tcw2.col(3));

		cv::Mat Ow1 = pRefKeyframe->GetCameraCenter();
		cv::Mat Ow2 = pCurKeyframe->GetCameraCenter();

		cv::Mat K = pRefKeyframe->K.clone();
		cv::Mat R1 = pRefKeyframe->GetRotation();
		cv::Mat t1 = pRefKeyframe->GetTranslation();
		cv::Mat R2 = pCurKeyframe->GetRotation();
		cv::Mat t2 = pCurKeyframe->GetTranslation();
		cv::Mat F12 = Utils::ComputeF12(R1, t1, R2, t2, K, K);

		// Triangulate each match
		std::vector<std::pair<size_t, size_t> > vMatchedIndices;
		int nMatch = SearchPoints::SearchForTriangulation(pRefKeyframe, pCurKeyframe, F12, vMatchedIndices);
		int nMap = 0;
		for (int ikp = 0; ikp < nMatch; ikp++)

		{
			const int& idx1 = vMatchedIndices[ikp].first;
			const int& idx2 = vMatchedIndices[ikp].second;

			const cv::KeyPoint& kp1 = pRefKeyframe->mvKeysUn[idx1];
			const cv::KeyPoint& kp2 = pCurKeyframe->mvKeysUn[idx2];

			// Check parallax between rays
			cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
			cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

			cv::Mat ray1 = Rwc1 * xn1;
			cv::Mat ray2 = Rwc2 * xn2;
			const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

			cv::Mat x3D;
			if (cosParallaxRays > 0 && cosParallaxRays < 0.9998)
			{
				// Linear Triangulation Method
				cv::Mat A(4, 4, CV_32F);
				A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
				A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
				A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
				A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

				cv::Mat w, u, vt;
				cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

				x3D = vt.row(3).t();

				if (x3D.at<float>(3) == 0)
					continue;

				// Euclidean coordinates
				x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

			}
			else
				continue; //No stereo and very low parallax

			cv::Mat x3Dt = x3D.t();

			//Check triangulation in front of cameras
			float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
			if (z1 <= 0)
				continue;

			float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
			if (z2 <= 0)
				continue;

			//Check reprojection error in first keyframe
			const float& sigmaSquare1 = pRefKeyframe->mvLevelSigma2[kp1.octave];
			const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
			const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
			const float invz1 = 1.0 / z1;

			float u1 = fx1 * x1 * invz1 + cx1;
			float v1 = fy1 * y1 * invz1 + cy1;
			float errX1 = u1 - kp1.pt.x;
			float errY1 = v1 - kp1.pt.y;
			float err1 = errX1 * errX1 + errY1 * errY1;
			/*if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
				continue;*/

				//Check reprojection error in second keyframe
			const float sigmaSquare2 = pCurKeyframe->mvLevelSigma2[kp2.octave];
			const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
			const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
			const float invz2 = 1.0 / z2;
			float u2 = fx2 * x2 * invz2 + cx2;
			float v2 = fy2 * y2 * invz2 + cy2;
			float errX2 = u2 - kp2.pt.x;
			float errY2 = v2 - kp2.pt.y;
			float err2 = (errX2 * errX2 + errY2 * errY2);
			/*if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
				continue;*/
				//std::cout << err1 << " " << err2 << std::endl;

			if (err1 > 4.0 || err2 > 4.0)
				continue;

			//Check scale consistency
			cv::Mat normal1 = x3D - Ow1;
			float dist1 = cv::norm(normal1);

			cv::Mat normal2 = x3D - Ow2;
			float dist2 = cv::norm(normal2);

			if (dist1 == 0 || dist2 == 0)
				continue;

			//// Triangulation is succesfull
			MapPoint* pMP = new MapPoint(x3D, pCurKeyframe, pMap, ts);
			pRefKeyframe->AddMapPoint(pMP, idx1);
			pCurKeyframe->AddMapPoint(pMP, idx2);
			pMP->AddObservation(pRefKeyframe, idx1);
			pMP->AddObservation(pCurKeyframe, idx2);
			pMP->ComputeDistinctiveDescriptors();
			pMP->UpdateNormalAndDepth();
			
			pCurFrame->mvpMapPoints[idx2] = pMP;
			pCurFrame->mvbOutliers[idx2] = false;

			pMap->AddMapPoint(pMP);
			pMap->mlpNewMPs.push_back(pMP);
			nMap++;
		}//for
		std::cout << "OXR create with prev frame " << nMap <<" "<<nMatch << std::endl;
	}

	KeyFrame* prevKF = nullptr;
	void Tracker::TrackWithKnownPose(ThreadPool::ThreadPool* pool, SLAM* system, int id, std::string user, double ts) {
		auto pUser = system->GetUser(user);
		if (!pUser)
			return;

		//ts
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long tts = start.time_since_epoch().count();

		//프레임을 만들고 항상 리로컬라이제이션을 함.
		auto cam = pUser->mpCamera;
		auto map = pUser->mpMap;

		WebAPI API("143.248.6.143", 35005);
		std::stringstream ss;
		ss << "/Load?keyword=OXR::IMAGE" << "&id=" << id << "&src=" << user;
		auto res = API.Send(ss.str(), "");
		int n = res.size();
		cv::Mat temp = cv::Mat(n, 1, CV_8UC1, (void*)res.data());
		cv::Mat img = cv::imdecode(temp, cv::IMREAD_COLOR);
		pUser->ImageDatas.Update(id, img);

		//매핑 모드
		//맵 생성

		//트래킹 모드
		//일단 리로컬 테스트
		cv::Mat T;
		Frame* frame = new Frame(img, cam, id, ts);
		auto mapState = map->GetState();
		auto userState = pUser->GetState();
		auto trackState = UserState::NotEstimated;
		std::cout << "???" << std::endl;
		if (pUser->mbDeviceTracking) {
			int nInliers = 0;
			if (mapState == MapState::Initialized) {
				bool bTrack = false;
				if (userState == UserState::NotEstimated || userState == UserState::Failed) {
					//global localization
					//set reference keyframe and last keyframe
					frame->reset_map_points();
					nInliers = Tracker::Relocalization(map, pUser, frame, system->mpFeatureTracker->min_descriptor_distance);
					if (nInliers >= 50)
					{
						bTrack = true;
						pUser->mnLastRelocFrameId = frame->mnFrameID;
						//trackState = UserState::Success;
						
					}
					std::cout << "Relocalization test = " << nInliers << std::endl;
				}
				else {
					//기존 트래킹처럼
				}
				//로컬 맵 최적화를 해야 함
			}

			pUser->SetState(trackState);
			if (trackState == UserState::Success) {

				//pose update
				cv::Mat T = frame->GetPose();
				pUser->UpdatePose(T, ts);
				pUser->PoseDatas.Update(id, T);

			}
		}
		if (pUser->mbMapping) {
			ss.str("");
			ss << "/Load?keyword=OXR::POSE" << "&id=" << id << "&src=" << user;
			res = API.Send(ss.str(), "");
			n = res.size();
			T = cv::Mat(4, 4, CV_32FC1, (void*)res.data());
			frame->SetPose(T);
			pUser->UpdatePose(T, ts);

			int nInliers = 0;
			if (mapState == MapState::NoImages) {
				//set reference frame
				map->SetState(MapState::NotInitialized);
				system->mpInitializer->Init(frame);
				return;
			}

			if (mapState == MapState::NotInitialized) {
				//initialization
				auto res = system->mpInitializer->InitializeOXR(frame, map);
				map->SetState(res);
				if (res == MapState::Initialized) {
					trackState = UserState::Success;
					auto kf1 = system->mpInitializer->mpInitKeyFrame1;
					auto kf2 = system->mpInitializer->mpInitKeyFrame2;

					kf1->sourceName = user;
					kf2->sourceName = user;

					map->mvpKeyFrameOrigins.push_back(kf1);
					pUser->mpRefKF = kf2;
					pUser->mnLastKeyFrameID = frame->mnFrameID;

					pool->EnqueueJob(LocalMapper::OXRMapping, pool, system, map, kf1);
					pool->EnqueueJob(LocalMapper::OXRMapping, pool, system, map, kf2);

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
				else {
					std::cout << "map initailization failed" << std::endl;
				}
			}
			else if (mapState == MapState::Initialized) {
				bool bTrack = false;
				if (userState == UserState::NotEstimated || userState == UserState::Failed) {
					//global localization
					//set reference keyframe and last keyframe
					frame->reset_map_points();
					//std::cout<<system->mpFeatureTracker->min_descriptor_distance
					nInliers = Tracker::Relocalization(map, pUser, frame, system->mpFeatureTracker->min_descriptor_distance);
					if (nInliers >= 50)
					{
						bTrack = true;
						pUser->mnLastRelocFrameId = frame->mnFrameID;
						//trackState = UserState::Success;
					}
				}
				else {
					if (userState == UserState::Success) {
						auto prev = pUser->prevFrame;
						prev->check_replaced_map_points();
						frame->reset_map_points();
						auto thMaxDesc = system->mpFeatureTracker->max_descriptor_distance;
						auto thMinDesc = system->mpFeatureTracker->min_descriptor_distance;
						int res = SearchPoints::SearchFrameByProjection(prev, frame, thMaxDesc, thMinDesc);
						if (res < 10) {
							frame->reset_map_points();
							res = SearchPoints::SearchFrameByProjection(prev, frame, thMaxDesc, thMinDesc, 30.0);
						}
						if (res < 10) {
							std::cout << "Matching prev fail!!! " << res << std::endl;
						}
						//std::cout<<"prev matching = "<<res<<std::endl;
						//local map matching
						LocalMap* pLocalMap = new LocalCovisibilityMap();
						pLocalMap->UpdateLocalMap(pUser, frame);
						//update visible
						int nMatch = Tracker::UpdateVisiblePoints(frame, pLocalMap->mvpLocalMPs, pLocalMap->mvpLocalTPs);
						int a = 0;
						if (nMatch > 0) {
							float thRadius = 1.0;
							if (frame->mnFrameID < pUser->mnLastRelocFrameId + 2)
								thRadius = 5.0;
							SearchPoints::SearchMapByProjection(frame, pLocalMap->mvpLocalMPs, pLocalMap->mvpLocalTPs, thMaxDesc, thMinDesc, thRadius);
						}
						nInliers = Tracker::UpdateFoundPoints(frame);
						if (nInliers > 10)
							bTrack = true;

						std::cout << "local map match = " << " " << nInliers << " " << res << std::endl;
						delete pLocalMap;
					}
				}
				if (!bTrack) {
					std::cout << "Tracking Fail Map Init" << std::endl;
					//trackState = UserState::Failed;

					//mapState = MapState::NotInitialized;
					//map->SetState(MapState::NotInitialized);
					nInliers = 0;
				}
				if (bTrack) {
					trackState = UserState::Success;
					//Segmentator::RequestObjectDetection(pUser->userName, frame->mnFrameID);
					std::cout << "Tracking Success" << std::endl;
				}
					
				{
					KeyFrame* pKF = new KeyFrame(frame, map);
					pKF->ComputeBoW();
					if (prevKF) {
						CreatePointsOXR(prevKF, pKF, frame, map);
					}
					prevKF = pKF;

					pKF->sourceName = pUser->userName;
					pKF->mbSendLocalMap = pUser->mbBaseLocalMap;
					pUser->mpRefKF = pKF;
					pUser->mnLastKeyFrameID = frame->mnFrameID;
					pool->EnqueueJob(LocalMapper::OXRMapping, pool, system, map, pKF);
					map->SetNotStop(false);
					pUser->KeyFrames.Update(frame->mnFrameID, pKF);
				}
			}

		}
		
		//frame.ComputeBoW();
		//int tempID = user->mnPrevFrameID;
		//pUser->ImageDatas.Update(id, temp);
		pUser->mnPrevFrameID = pUser->mnCurrFrameID.load();
		pUser->mnCurrFrameID = frame->mnFrameID;

		if (mapState == MapState::Initialized && pUser->prevFrame)
			delete pUser->prevFrame;
		pUser->prevFrame = frame;
		pUser->mbProgress = false;

		////visualization
		if (mapState == MapState::Initialized && pUser->GetVisID() <= 3 && userState != UserState::NotEstimated) {

			cv::Mat R = frame->GetRotation();
			cv::Mat t = frame->GetTranslation();
			cv::Mat K = pUser->GetCameraMatrix();

			cv::Scalar color = Segmentator::mvObjectLabelColors[pUser->GetVisID() + 1];

			for (int i = 0; i < frame->mvKeys.size(); i++) {
				auto pMP = frame->mvpMapPoints[i];
				//cv::Scalar color = cv::Scalar(255, 0, 255);
				int r = 2;
				if (pMP && !pMP->isBad())
				{
					cv::circle(img, frame->mvKeys[i].pt, r, color, -1);

					cv::Mat x3D = pMP->GetWorldPos();
					cv::Mat proj = K * (R * x3D + t);
					float d = proj.at<float>(2);
					cv::Point2f pt(proj.at<float>(0) / d, proj.at<float>(1) / d);
					cv::circle(img, pt, r, cv::Scalar(255, 0, 255), -1);
				}
			}
			system->VisualizeImage(pUser->mapName, img, pUser->GetVisID() + 4);
		}
		pUser->mnDebugTrack--;
		pUser->mnUsed--;
	}

	void Tracker::Track(ThreadPool::ThreadPool* pool, SLAM* system, int id, std::string user, Frame* frame, const cv::Mat& img, double ts) {
		auto pUser = system->GetUser(user);
		if (!pUser)
			return;
		if (pUser->mbProgress)
			return;
		if (id < pUser->mnPrevFrameID)
			return;
		pUser->mnUsed++;
		pUser->mnDebugTrack++;
		pUser->mbProgress = true;
		auto cam = pUser->mpCamera;
		auto map = pUser->mpMap;
		
		//bool bSimulation = false;//url.find("SimImage") != std::string::npos;
		//bool bClient = false;
		////simulation check
		//if (bSimulation){
		//	if (map->GetState() == MapState::NotInitialized) {
		//		pUser->mbProgress = false;
		//		pUser->mnUsed--;
		//		pUser->mnDebugTrack--;
		//		return;
		//	}
		//	bClient = user.find("client_1") != std::string::npos;
		//}

		//time analysis
		float t_local = 1000.0;
		float t_send = 1000.0;
		float t_prev = 1000.0;
		float t_init = 1000.0;
		float t_frame = 1000.0;
		LocalMap* pLocalMap = new LocalCovisibilityMap();

		//std::cout << "Frame = " << user->userName <<" "<<id<< "=start!!" << std::endl;
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		auto strTimeStamp = Utils::GetTimeStamp();

		////receive image
		std::chrono::high_resolution_clock::time_point received = std::chrono::high_resolution_clock::now();
		
		/////KP 여기도 플래그 추가하기
		//Frame* frame = nullptr;
		std::chrono::high_resolution_clock::time_point t_frame_s = std::chrono::high_resolution_clock::now();
		//frame = new Frame(img, cam, id, ts);
		std::chrono::high_resolution_clock::time_point t_frame_e = std::chrono::high_resolution_clock::now();
		auto du_frame = std::chrono::duration_cast<std::chrono::milliseconds>(t_frame_e - t_frame_s).count();
		t_frame = du_frame / 1000.0;
		
		auto mapState = map->GetState();
		auto userState = pUser->GetState();
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

				kf1->sourceName = user;
				kf2->sourceName = user;

				map->mvpKeyFrameOrigins.push_back(kf1);
				/*user->mapKeyFrames[kf1->mnId] = kf1;
				user->mapKeyFrames[kf2->mnId] = kf2;*/
				pUser->mpRefKF = kf2;
				pUser->mnLastKeyFrameID = frame->mnFrameID;
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
				nInliers = Tracker::Relocalization(map, pUser, frame, system->mpFeatureTracker->min_descriptor_distance);
				if (nInliers >= 50)
				{
					bTrack = true;
					pUser->mnLastRelocFrameId = frame->mnFrameID;
					//trackState = UserState::Success;
				}
				std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
				auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				float t_test1 = du_test1 / 1000.0;
				
				int N = system->GetConnectedDevice();
				//system->ProcessingTime.Get("reloc")[N]->add(t_test1);
			}
			else {
				if (userState == UserState::Success) {
					
					//std::cout << "Tracker::Start" << std::endl;
					//auto f_ref = user->mapFrames[user->mnPrevFrameID];
					auto prevFrame = pUser->prevFrame;
					prevFrame->check_replaced_map_points();

					cv::Mat Tpredict;
					if (pUser->mbIMU) {
						auto Rgyro = pUser->GetGyro();
						cv::Mat Tgyro = cv::Mat::eye(4, 4, CV_32FC1);
						Rgyro.copyTo(Tgyro.rowRange(0, 3).colRange(0, 3));
						Tpredict = Tgyro * pUser->GetPose();
					}
					else {
						Tpredict = pUser->PredictPose();
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
						/*std::cout << "track with reference frame :: start" << std::endl;
						std::cout << "track with reference frame :: end" << std::endl;*/
					}
					else {
						
					}
				}
				
			}
			
			if (bTrack) {
				std::chrono::high_resolution_clock::time_point t_local_start = std::chrono::high_resolution_clock::now();
				nInliers = Tracker::TrackWithLocalMap(pLocalMap, pUser, frame, system->mpFeatureTracker->max_descriptor_distance, system->mpFeatureTracker->min_descriptor_distance);
				std::chrono::high_resolution_clock::time_point t_local_end = std::chrono::high_resolution_clock::now();
				auto du_local = std::chrono::duration_cast<std::chrono::milliseconds>(t_local_end - t_local_start).count();
				t_local = du_local / 1000.0;
				if (frame->mnFrameID < pUser->mnLastRelocFrameId + 30 && nInliers < 50) {
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
			
			cv::Mat T = frame->GetPose().clone();
			
			/*if (pUser->mbBaseLocalMap && pUser->mpRefKF) {
				pool->EnqueueJob(Tracker::SendLocalMap, system, user, id);
			}*/
			//로컬 맵 키프레임 전송 중 에러.
			//키프레임의 포즈까지 전송.
			
		}
		
		pUser->SetState(trackState);
		if (trackState == UserState::Success) {
			
			////오브젝트 검출 요청
			//Segmentator::RequestObjectDetection(user->userName, frame->mnFrameID);
			
			//pose update
			cv::Mat T = frame->GetPose();
			cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
			cv::Mat t = T.rowRange(0, 3).col(3);
			R.convertTo(R, CV_64FC1);
			t.convertTo(t, CV_64FC1);
			pUser->mpKalmanFilter->fillMeasurements(t,R);
			pUser->mpKalmanFilter->updateKalmanFilter(t,R);
			pUser->UpdatePose(T, ts);
			pUser->PoseDatas.Update(id, T);

			pUser->MapServerTrajectories.Update(id, T.inv());
			//check keyframe
			if (pUser->mbMapping && pUser->mpRefKF) {
				if (Tracker::NeedNewKeyFrame(map, system->mpLocalMapper, frame, pUser->mpRefKF, nInliers, pUser->mnLastKeyFrameID.load(), pUser->mnLastRelocFrameId.load())) {
					Tracker::CreateNewKeyFrame(pool, system, map, system->mpLocalMapper, frame, pUser);
					Segmentator::RequestSegmentation(pUser->userName, frame->mnFrameID);
					Segmentator::RequestObjectDetection(pUser->userName, frame->mnFrameID);
					system->RequestTime.Update(id, std::chrono::high_resolution_clock::now());

					/*{
						auto kfs = std::vector<KeyFrame*>(pLocalMap->mvpLocalKFs.begin(), pLocalMap->mvpLocalKFs.end());
						float fx = frame->fx;
						float fy = frame->fy;
						float cx = frame->cx;
						float cy = frame->cy;
						pool->EnqueueJob(Tracker::SendFrameInformationForRecon, system, id, pUser->userName, T, fx, fy, cx, cy, kfs);
					}*/

				}
			}
			
		}
		else
			pUser->PoseDatas.Update(id, cv::Mat());
		
		//int tempID = user->mnPrevFrameID;
		
		pUser->mnPrevFrameID = pUser->mnCurrFrameID.load();
		pUser->mnCurrFrameID = frame->mnFrameID;

		if (mapState == MapState::Initialized && pUser->prevFrame)
			delete pUser->prevFrame;
		pUser->prevFrame = frame;
		pUser->mbProgress = false;
		
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(received - start).count();
		float t_test1 = du_test1 / 1000.0;

		auto du_test2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - received).count();
		float t_test2 = du_test2 / 1000.0;

		delete pLocalMap;
		
		//if (bSimulation) {
		//	int N = system->GetConnectedDevice();
		//	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(start.time_since_epoch()).count();
		//	std::stringstream ss;
		//	ss <<strTimeStamp.str()<<" "<< id << " " << N << " " << t_test1 << " " << t_test2 << std::endl;
		//	system->EvaluationLatency.push_back(ss.str());
		//	
		//	//system->ProcessingTime.Get("download")[N]->add(t_test1);
		//	//system->ProcessingTime.Get("tracking")[N]->add(t_test2);
		//	//std::cout << "simul test = " <<N<<"== "<< t_test1 << " " << t_test2 << std::endl;
		//}
			

		////데이터 측정 코드
		/*int N = system->GetConnectedDevice();
		system->ProcessingTime.Get("download")[N]->add(t_test1);
		system->ProcessingTime.Get("tracking")[N]->add(t_test2);
		if (mapState == MapState::Initialized && !pUser->mbMapping) {
			int ntemp = userState == UserState::Success ? 1 : 0;
			system->SuccessRatio.Get("skipframe")[pUser->mnSkip]->increase(ntemp);

			if (pUser->mbAsyncTest) {
				system->SuccessRatio.Get("async")[pUser->mnQuality]->increase(ntemp);
			}

		}*/
		//system->UpdateTrackingTime(t_test1);
		////데이터 측정 코드

		////visualization
		//if (mapState == MapState::Initialized  && pUser->GetVisID() <= 3 && userState != UserState::NotEstimated) {

		//	cv::Mat R = frame->GetRotation();
		//	cv::Mat t = frame->GetTranslation();
		//	cv::Mat K = pUser->GetCameraMatrix();

		//	cv::Scalar color = Segmentator::mvObjectLabelColors[pUser->GetVisID()+1];

		//	for (int i = 0; i < frame->mvKeys.size(); i++) {
		//		auto pMP = frame->mvpMapPoints[i];
		//		//cv::Scalar color = cv::Scalar(255, 0, 255);
		//		int r = 2;
		//		if (pMP && !pMP->isBad())
		//		{
		//			cv::circle(img, frame->mvKeys[i].pt, r, color, -1);

		//			cv::Mat x3D = pMP->GetWorldPos();	
		//			cv::Mat proj= K*(R*x3D + t);
		//			float d = proj.at<float>(2);
		//			cv::Point2f pt(proj.at<float>(0) / d, proj.at<float>(1) / d);
		//			if(pMP->mnObjectID == 100)
		//				cv::circle(img, pt, r+3, cv::Scalar(255, 255, 0));
		//			else
		//				cv::circle(img, pt, r, cv::Scalar(255,0,255), -1);
		//		}
		//	}

		//	system->VisualizeImage(pUser->mapName, img, pUser->GetVisID()+4); 
		//}
		pUser->mnDebugTrack--;
		pUser->mnUsed--;
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
		if (nMatch > 0) {
			float thRadius = 1.0;
			if (cur->mnFrameID < user->mnLastRelocFrameId + 2)
				thRadius = 5.0;

			int a = SearchPoints::SearchMapByProjection(cur, pLocalMap->mvpLocalMPs, pLocalMap->mvpLocalTPs, thMaxDesc, thMinDesc, thRadius);
			Optimizer::PoseOptimization(cur);

			////user update
			auto setPrevKFs = user->mSetLocalKeyFrames.Get();
			user->mSetLocalKeyFrames.Set(pLocalMap->mspLocalKFs);

			for (auto iter = setPrevKFs.begin(), iend = setPrevKFs.end(); iter != iend; iter++) {
				auto pKF = *iter;
				pKF->mnConnectedDevices--;
			}
			for (auto iter = pLocalMap->mspLocalKFs.begin(), iend = pLocalMap->mspLocalKFs.end(); iter != iend; iter++) {
				auto pKF = *iter;
				pKF->mnConnectedDevices++;
			}
			/*for (int i = 0; i<cur->N; i++)
			{
				auto pMPi = cur->mvpMapPoints[i];
				if(pMPi && !pMPi->isBad() && !cur->mvbOutliers[i]){
					if (!pMPi->mSetConnected.Count(user))
						pMPi->mSetConnected.Update(user);
					if (!user->mSetMapPoints.Count(pMPi)) {
						user->mSetMapPoints.Update(pMPi);
					}
				}
			}*/
			////user update
		}
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
				//std::cout << "relocalization=" << i << "=final::ngood=" << nGood << std::endl;
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
				//else {
				//	//사람과 오브젝트 처리용
				//	cur->mvpMapPoints[i] = nullptr;
				//}
			}
		}
		return nres;
	}

	bool Tracker::NeedNewKeyFrame(Map* map, LocalMapper* mapper, Frame* cur, KeyFrame* ref, int nMatchesInliers, int nLastKeyFrameId, int nLastRelocFrameID, bool bOnlyTracking, int nMaxFrames, int nMinFrames)
	{
		if (bOnlyTracking)
			return false;

		// If Local Mapping is freezed by a Loop Closure do not insert keyframes
		if (map->isStopped() || map->stopRequested()){
			return false;
		}
		const int nKFs = map->GetNumKeyFrames();

		// Do not insert keyframes if not enough frames have passed from last relocalisation
		if (cur->mnFrameID<nLastRelocFrameID + nMaxFrames && nKFs>nMaxFrames){
			return false;
		}

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
		pKF->sourceName = user->userName;
		pKF->mbSendLocalMap = user->mbBaseLocalMap;
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
	//키프레임 아이디와 포즈, 현재 이미지의 아이디와 이미지 전송(이것을 이후 키프레임 연경)
	void Tracker::SendFrameInformationForRecon(EdgeSLAM::SLAM* system, int id, std::string userName, const cv::Mat& T, float fx, float fy, float cx, float cy, const std::vector<KeyFrame*>& kfs) {

		//이미지, 포즈, fx, fy 전송
		////MP와 인접 MP 정보 전송
		WebAPI API("143.248.6.143", 35005);
		std::stringstream ss;

		//현재 프레임의 자세 및 인트린직 정보와 인접 키프레임의 아이디와 포즈 전송.

		//레퍼런스와 6개의 키프레임
		cv::Mat data = cv::Mat::zeros(5000, 1, CV_32FC1); //inlier, pose + point2f, octave, angle, point3f
														//12+4+7+1(연결 KF 수)

		//프레임의 포즈와 인트린직
		int nDataIdx = 0;
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
		data.at<float>(nDataIdx++) = fx;
		data.at<float>(nDataIdx++) = fy;
		data.at<float>(nDataIdx++) = cx;
		data.at<float>(nDataIdx++) = cy;

		nDataIdx = 17;
		int nKF = 0;
		for (int i = 0; i < kfs.size(); i++) {
			auto pKF = kfs[i];
			//포즈
			if (!pKF || pKF->isBad()) {
				continue;
			}
			
			cv::Mat kfPose = pKF->GetPose();
			data.at<float>(nDataIdx++) = (float)pKF->mnFrameId;
			data.at<float>(nDataIdx++) = kfPose.at<float>(0, 0);
			data.at<float>(nDataIdx++) = kfPose.at<float>(0, 1);
			data.at<float>(nDataIdx++) = kfPose.at<float>(0, 2);
			data.at<float>(nDataIdx++) = kfPose.at<float>(1, 0);
			data.at<float>(nDataIdx++) = kfPose.at<float>(1, 1);
			data.at<float>(nDataIdx++) = kfPose.at<float>(1, 2);
			data.at<float>(nDataIdx++) = kfPose.at<float>(2, 0);
			data.at<float>(nDataIdx++) = kfPose.at<float>(2, 1);
			data.at<float>(nDataIdx++) = kfPose.at<float>(2, 2);
			data.at<float>(nDataIdx++) = kfPose.at<float>(0, 3);
			data.at<float>(nDataIdx++) = kfPose.at<float>(1, 3);
			data.at<float>(nDataIdx++) = kfPose.at<float>(2, 3);

			nKF++;
			if (nKF == 7)
				break;
		}
		data.at<float>(16) = (float)nKF;

		ss << "/Store?keyword=NewFrameForRecon&id=" << id << "&src=" << userName;
		auto res = API.Send(ss.str(), data.data, data.rows * sizeof(float));
	}

	void Tracker::SendDeviceTrackingData(SLAM* system, std::string userName, const cv::Mat& data, int id, double ts) {
		{
			WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
			std::stringstream ss;
			ss << "/Store?keyword=ReferenceFrame&id=" << id << "&src=" << userName <<"&ts="<<std::fixed<< std::setprecision(6) <<ts<< "&type2=" << userName;
			auto res = mpAPI->Send(ss.str(), data.data, data.rows * sizeof(float));
			delete mpAPI;
		}
	}

	void Tracker::SendLocalMap(EdgeSLAM::SLAM* SLAM, std::string user, int id) {
		auto pUser = SLAM->GetUser(user);
		if (!pUser)
			return;
		pUser->mnUsed++;

		auto spLocalKFs = pUser->mSetLocalKeyFrames.Get();
		auto vpLocalKFs = std::vector<EdgeSLAM::KeyFrame*>(spLocalKFs.begin(), spLocalKFs.end());
		std::set<EdgeSLAM::MapPoint*> spMPs;
		std::vector<EdgeSLAM::MapPoint*> vpLocalMPs;

		//pts에 맵포인트 id와 3차원 위치, 민, 맥스, 노말 추가
		//4 + 12 + 4 + 4 + 12 = 36바이트가 필요함.(디스크립터 제외), 디스크립터는 32바이트임.
		cv::Mat pts = cv::Mat::zeros(0, 1, CV_32FC1);
		cv::Mat desc = cv::Mat::zeros(0, 1, CV_8UC1);

		int nInput = 0;

		for (std::vector<EdgeSLAM::KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			EdgeSLAM::KeyFrame* pKFi = *itKF;
			if (!pKFi)
				continue;
			const std::vector<EdgeSLAM::MapPoint*> vpMPs = pKFi->GetMapPointMatches();

			int nInputTemp = 0;
			for (std::vector<EdgeSLAM::MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
				EdgeSLAM::MapPoint* pMPi = *itMP;
				if (!pMPi || pMPi->isBad() || spMPs.count(pMPi))
					continue;
				vpLocalMPs.push_back(pMPi);
				spMPs.insert(pMPi);

				///추가 데이터
				float id = (float)pMPi->mnId;
				float minDist = pMPi->GetMinDistanceInvariance() / 0.8;
				float maxDist = pMPi->GetMaxDistanceInvariance() / 1.2;
				cv::Mat temp = cv::Mat::zeros(3, 1, CV_32FC1);
				temp.at<float>(0) = id;
				temp.at<float>(1) = minDist;
				temp.at<float>(2) = maxDist;

				pts.push_back(temp);
				pts.push_back(pMPi->GetWorldPos());
				pts.push_back(pMPi->GetNormal());
				desc.push_back(pMPi->GetDescriptor().t());
			}
		}
		pUser->mnUsed--;

		cv::Mat converted_desc = cv::Mat::zeros(vpLocalMPs.size() * 8, 1, CV_32FC1);
		std::memcpy(converted_desc.data, desc.data, desc.rows);
		pts.push_back(converted_desc);
		
		{
			WebAPI API("143.248.6.143", 35005);
			std::stringstream ss;
			ss << "/Store?keyword=UpdatedLocalMap&id=" << id << "&src=" << user;
			auto res = API.Send(ss.str(), pts.data, pts.rows * sizeof(float));
		}
	}
}