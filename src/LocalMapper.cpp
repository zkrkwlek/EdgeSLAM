#include <LocalMapper.h>
#include <SLAM.h>
#include <LoopCloser.h>
#include <KeyFrame.h>
#include <Frame.h>
#include <MapPoint.h>
#include <Map.h>
#include <FeatureTracker.h>
#include <SearchPoints.h>
#include <Optimizer.h>
#include <Plane.h>
#include <Utils.h>
#include <chrono>


namespace EdgeSLAM {
	LocalMapper::LocalMapper()
	{}
	LocalMapper::~LocalMapper(){}

	void LocalMapper::ProcessMapping(ThreadPool::ThreadPool* pool, SLAM* system, Map* map, KeyFrame* targetKF) {

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		
		map->mnNumMappingFrames++;
		auto pMapper = system->mpLocalMapper;
		pMapper->ProcessNewKeyFrame(map, targetKF);
		pMapper->MapPointCulling(map, targetKF);
		pMapper->CreateNewMapPoints(map, targetKF);
		if(map->mnNumMappingFrames == 1)
			pMapper->SearchInNeighbors(map, targetKF);
		
		//pool->EnqueueJob(PlaneProcessor::EstimateLocalMapPlanes, system, map, targetKF);
		
		map->mbAbortBA = false;
		if (map->mnNumMappingFrames == 1 && !map->stopRequested())
		{
			if (map->GetNumKeyFrames() > 2) {
				//std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
				Optimizer::LocalBundleAdjustment(targetKF, &map->mbAbortBA, map);
				/*std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
				auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				float t_test1 = du_test1 / 1000.0;
				std::cout << "BA = " << t_test1 << std::endl;*/
			}
			pMapper->KeyFrameCulling(map, targetKF);
		}
		
		pool->EnqueueJob(LoopCloser::ProcessLoopClosing, system, map, targetKF);
		map->mnNumMappingFrames--;

		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		float t_test1 = du_test1 / 1000.0;
		int N = system->GetConnectedDevice();
		system->ProcessingTime.Get(N)["mapping"]->add(t_test1);
	}
	void LocalMapper::ProcessNewKeyFrame(Map* map, KeyFrame* targetKF) {
		
		targetKF->ComputeBoW();
		const std::vector<MapPoint*> vpMapPointMatches = targetKF->GetMapPointMatches();

		for (size_t i = 0; i<vpMapPointMatches.size(); i++)
		{
			MapPoint* pMP = vpMapPointMatches[i];
			if (pMP)
			{
				if (!pMP->isBad())
				{
					if (!pMP->IsInKeyFrame(targetKF))
					{
						pMP->AddObservation(targetKF, i);
						pMP->UpdateNormalAndDepth();
						pMP->ComputeDistinctiveDescriptors();
					}
					else 
					{
						// this can only happen for new stereo points inserted by the Tracking
						map->mlpNewMPs.push_back(pMP);
					}
				}
			}
		}

		// Update links in the Covisibility Graph
		targetKF->UpdateConnections();

		// Insert Keyframe in Map
		map->AddKeyFrame(targetKF);
	}

	void LocalMapper::MapPointCulling(Map* map, KeyFrame* targetKF)
	{
		// Check Recent Added MapPoints
		std::list<MapPoint*>::iterator lit = map->mlpNewMPs.begin();
		const unsigned long int nCurrentKFid = targetKF->mnId;

		const int cnThObs = 2;

		while (lit != map->mlpNewMPs.end())
		{
			MapPoint* pMP = *lit;
			if (pMP->isBad())
			{
				lit = map->mlpNewMPs.erase(lit);
			}
			else if (pMP->GetFoundRatio()<0.25f)
			{
				pMP->SetBadFlag();
				lit = map->mlpNewMPs.erase(lit);
			}
			else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 2 && pMP->Observations() <= cnThObs)
			{
				pMP->SetBadFlag();
				lit = map->mlpNewMPs.erase(lit);
			}
			else if (((int)nCurrentKFid - (int)pMP->mnFirstKFid) >= 3)
				lit = map->mlpNewMPs.erase(lit);
			else
				lit++;
		}
	}

	void LocalMapper::CreateNewMapPoints(Map* map, KeyFrame* targetKF)
	{
		
		// Retrieve neighbor keyframes in covisibility graph
		int nn = 20;
		const std::vector<KeyFrame*> vpNeighKFs = targetKF->GetBestCovisibilityKeyFrames(nn);
		//0.6, false
		//ORBmatcher matcher(0.6, false);

		cv::Mat Rcw1 = targetKF->GetRotation();
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat tcw1 = targetKF->GetTranslation();
		cv::Mat Tcw1(3, 4, CV_32F);
		Rcw1.copyTo(Tcw1.colRange(0, 3));
		tcw1.copyTo(Tcw1.col(3));
		cv::Mat Ow1 = targetKF->GetCameraCenter();

		const float &fx1 = targetKF->fx;
		const float &fy1 = targetKF->fy;
		const float &cx1 = targetKF->cx;
		const float &cy1 = targetKF->cy;
		const float &invfx1 = targetKF->invfx;
		const float &invfy1 = targetKF->invfy;

		const float ratioFactor = 1.5f*targetKF->mfScaleFactor;

		int nnew = 0;

		cv::Mat R1 = targetKF->GetRotation();
		cv::Mat t1 = targetKF->GetTranslation();

		// Search matches with epipolar restriction and triangulate
		for (size_t i = 0; i<vpNeighKFs.size(); i++)
		{
			if (i>0 && map->mnNumMappingFrames>1)
				return;

			KeyFrame* pKF2 = vpNeighKFs[i];

			// Check first that baseline is not too short
			cv::Mat Ow2 = pKF2->GetCameraCenter();
			cv::Mat vBaseline = Ow2 - Ow1;
			const float baseline = cv::norm(vBaseline);

			const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
			const float ratioBaselineDepth = baseline / medianDepthKF2;

			if (ratioBaselineDepth<0.01)
				continue;

			// Compute Fundamental Matrix
			cv::Mat R2 = pKF2->GetRotation();
			cv::Mat t2 = pKF2->GetTranslation();
			cv::Mat F12 = Utils::ComputeF12(R1, t1, R2, t2, targetKF->K, pKF2->K);

			// Search matches that fullfil epipolar constraint
			std::vector<std::pair<size_t, size_t> > vMatchedIndices;

			SearchPoints::SearchForTriangulation(targetKF, pKF2, F12, vMatchedIndices);
			
			cv::Mat Rcw2 = pKF2->GetRotation();
			cv::Mat Rwc2 = Rcw2.t();
			cv::Mat tcw2 = pKF2->GetTranslation();
			cv::Mat Tcw2(3, 4, CV_32F);
			Rcw2.copyTo(Tcw2.colRange(0, 3));
			tcw2.copyTo(Tcw2.col(3));

			const float &fx2 = pKF2->fx;
			const float &fy2 = pKF2->fy;
			const float &cx2 = pKF2->cx;
			const float &cy2 = pKF2->cy;
			const float &invfx2 = pKF2->invfx;
			const float &invfy2 = pKF2->invfy;

			// Triangulate each match
			const int nmatches = vMatchedIndices.size();
			for (int ikp = 0; ikp<nmatches; ikp++)

			{
				const int &idx1 = vMatchedIndices[ikp].first;
				const int &idx2 = vMatchedIndices[ikp].second;

				const cv::KeyPoint &kp1 = targetKF->mvKeysUn[idx1];
				const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

				// Check parallax between rays
				cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1)*invfx1, (kp1.pt.y - cy1)*invfy1, 1.0);
				cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2)*invfx2, (kp2.pt.y - cy2)*invfy2, 1.0);

				cv::Mat ray1 = Rwc1*xn1;
				cv::Mat ray2 = Rwc2*xn2;
				const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1)*cv::norm(ray2));

				cv::Mat x3D;
				if (cosParallaxRays>0 && cosParallaxRays<0.9998)
				{
					// Linear Triangulation Method
					cv::Mat A(4, 4, CV_32F);
					A.row(0) = xn1.at<float>(0)*Tcw1.row(2) - Tcw1.row(0);
					A.row(1) = xn1.at<float>(1)*Tcw1.row(2) - Tcw1.row(1);
					A.row(2) = xn2.at<float>(0)*Tcw2.row(2) - Tcw2.row(0);
					A.row(3) = xn2.at<float>(1)*Tcw2.row(2) - Tcw2.row(1);

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
				const float &sigmaSquare1 = targetKF->mvLevelSigma2[kp1.octave];
				const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
				const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
				const float invz1 = 1.0 / z1;

				float u1 = fx1*x1*invz1 + cx1;
				float v1 = fy1*y1*invz1 + cy1;
				float errX1 = u1 - kp1.pt.x;
				float errY1 = v1 - kp1.pt.y;
				if ((errX1*errX1 + errY1*errY1)>5.991*sigmaSquare1)
					continue;

				//Check reprojection error in second keyframe
				const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
				const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
				const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
				const float invz2 = 1.0 / z2;
				float u2 = fx2*x2*invz2 + cx2;
				float v2 = fy2*y2*invz2 + cy2;
				float errX2 = u2 - kp2.pt.x;
				float errY2 = v2 - kp2.pt.y;
				if ((errX2*errX2 + errY2*errY2)>5.991*sigmaSquare2)
					continue;

				//Check scale consistency
				cv::Mat normal1 = x3D - Ow1;
				float dist1 = cv::norm(normal1);

				cv::Mat normal2 = x3D - Ow2;
				float dist2 = cv::norm(normal2);

				if (dist1 == 0 || dist2 == 0)
					continue;

				const float ratioDist = dist2 / dist1;
				const float ratioOctave = targetKF->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

				/*if(fabs(ratioDist-ratioOctave)>ratioFactor)
				continue;*/
				if (ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
					continue;

				// Triangulation is succesfull
				MapPoint* pMP = new MapPoint(x3D, targetKF, map);

				pMP->AddObservation(targetKF, idx1);
				pMP->AddObservation(pKF2, idx2);

				targetKF->AddMapPoint(pMP, idx1);
				pKF2->AddMapPoint(pMP, idx2);

				pMP->ComputeDistinctiveDescriptors();

				pMP->UpdateNormalAndDepth();

				map->AddMapPoint(pMP);
				map->mlpNewMPs.push_back(pMP);

				nnew++;
			}
		}
	}

	void LocalMapper::SearchInNeighbors(Map* map, KeyFrame* targetKF)
	{
		// Retrieve neighbor keyframes
		int nn = 20;
		const std::vector<KeyFrame*> vpNeighKFs = targetKF->GetBestCovisibilityKeyFrames(nn);
		std::vector<KeyFrame*> vpTargetKFs;
		for (std::vector<KeyFrame*>::const_iterator vit = vpNeighKFs.begin(), vend = vpNeighKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;
			if (pKFi->isBad() || pKFi->mnFuseTargetForKF == targetKF->mnId)
				continue;
			vpTargetKFs.push_back(pKFi);
			pKFi->mnFuseTargetForKF = targetKF->mnId;

			// Extend to some second neighbors
			const std::vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
			for (std::vector<KeyFrame*>::const_iterator vit2 = vpSecondNeighKFs.begin(), vend2 = vpSecondNeighKFs.end(); vit2 != vend2; vit2++)
			{
				KeyFrame* pKFi2 = *vit2;
				if (pKFi2->isBad() || pKFi2->mnFuseTargetForKF == targetKF->mnId || pKFi2->mnId == targetKF->mnId)
					continue;
				vpTargetKFs.push_back(pKFi2);
			}
		}


		// Search matches by projection from current KF in target KFs
		std::vector<MapPoint*> vpMapPointMatches = targetKF->GetMapPointMatches();
		for (std::vector<KeyFrame*>::iterator vit = vpTargetKFs.begin(), vend = vpTargetKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;
			SearchPoints::Fuse(pKFi, vpMapPointMatches);
		}

		// Search matches by projection from target KFs in current KF
		std::vector<MapPoint*> vpFuseCandidates;
		//vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

		for (std::vector<KeyFrame*>::iterator vitKF = vpTargetKFs.begin(), vendKF = vpTargetKFs.end(); vitKF != vendKF; vitKF++)
		{
			KeyFrame* pKFi = *vitKF;

			std::vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

			for (std::vector<MapPoint*>::iterator vitMP = vpMapPointsKFi.begin(), vendMP = vpMapPointsKFi.end(); vitMP != vendMP; vitMP++)
			{
				MapPoint* pMP = *vitMP;
				if (!pMP)
					continue;
				if (pMP->isBad() || pMP->mnFuseCandidateForKF == targetKF->mnId)
					continue;
				pMP->mnFuseCandidateForKF = targetKF->mnId;
				vpFuseCandidates.push_back(pMP);
			}
		}

		int nFused =  SearchPoints::Fuse(targetKF, vpFuseCandidates);
		// Update points
		vpMapPointMatches = targetKF->GetMapPointMatches();
		for (size_t i = 0, iend = vpMapPointMatches.size(); i<iend; i++)
		{
			MapPoint* pMP = vpMapPointMatches[i];
			if (pMP)
			{
				if (!pMP->isBad())
				{
					pMP->ComputeDistinctiveDescriptors();
					pMP->UpdateNormalAndDepth();
				}
			}
		}

		// Update connections in covisibility graph
		targetKF->UpdateConnections();
	}

	void LocalMapper::KeyFrameCulling(Map* map, KeyFrame* targetKF)
	{
		// Check redundant keyframes (only local keyframes)
		// A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
		// in at least other 3 keyframes (in the same or finer scale)
		// We only consider close stereo points
		std::vector<KeyFrame*> vpLocalKeyFrames = targetKF->GetVectorCovisibleKeyFrames();

		for (std::vector<KeyFrame*>::iterator vit = vpLocalKeyFrames.begin(), vend = vpLocalKeyFrames.end(); vit != vend; vit++)
		{
			KeyFrame* pKF = *vit;
			if (pKF->mnId == 0)
				continue;
			const std::vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

			int nObs = 3;
			const int thObs = nObs;
			int nRedundantObservations = 0;
			int nMPs = 0;
			for (size_t i = 0, iend = vpMapPoints.size(); i<iend; i++)
			{
				MapPoint* pMP = vpMapPoints[i];
				if (pMP)
				{
					if (!pMP->isBad())
					{
						nMPs++;
						if (pMP->Observations()>thObs)
						{
							const int &scaleLevel = pKF->mvKeysUn[i].octave;
							const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();
							int nObs = 0;
							for (std::map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
							{
								KeyFrame* pKFi = mit->first;
								if (pKFi == pKF)
									continue;
								const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

								if (scaleLeveli <= scaleLevel + 1)
								{
									nObs++;
									if (nObs >= thObs)
										break;
								}
							}
							if (nObs >= thObs)
							{
								nRedundantObservations++;
							}
						}
					}
				}
			}

			if (nRedundantObservations>0.9*nMPs)
				pKF->SetBadFlag();
		}
	}

	////thread process

	
	////thread process
}