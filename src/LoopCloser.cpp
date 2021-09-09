#include <LoopCloser.h>
#include <SLAM.h>
#include <Map.h>
#include <Tracker.h>
#include <LocalMapper.h>
#include <KeyFrameDB.h>
#include <MapPoint.h>
#include <KeyFrame.h>
#include <Sim3Solver.h>
#include <SearchPoints.h>
#include <FeatureTracker.h>
#include <Converter.h>
#include <Optimizer.h>
#include <windows.h>

namespace EdgeSLAM {
	LoopCloser::LoopCloser():mnCovisibilityConsistencyTh(3){}
	//LoopCloser::LoopCloser(ORBVocabulary* voc, bool bFixScale) {}
	LoopCloser::~LoopCloser(){}
	void LoopCloser::ProcessLoopClosing(SLAM* system, Map* map, KeyFrame* kf) {
		//update keyframe database
		auto pLoopCloser = system->mpLoopCloser;
		if (kf->mnId < 2)
			return;
		std::cout << "LoopClosing #" << map->mnNumLoopClosingFrames << " frames" << std::endl;
		if (map->mnNumLoopClosingFrames == 1) {
			std::cout << "Loop closing error!!!!!!!!!!!!!!" << std::endl;
			return;
		}
		map->mnNumLoopClosingFrames++;
		
		bool bSim3 = false;
		bool bDetect = pLoopCloser->DetectLoop(system, map, kf);

		/*if (bDetect)
		{
			std::cout << "ComputeSim3::start" << std::endl;
			bSim3 = pLoopCloser->ComputeSim3(system, map, kf);
			std::cout << "ComputeSim3::end" << std::endl;
		}
		if (bSim3) {
			std::cout << "CorrectLoop::start" << std::endl;
			pLoopCloser->CorrectLoop(system, map, kf);
			std::cout << "CorrectLoop::start" << std::endl;
		}*/
		map->mnNumLoopClosingFrames--;
	}
	bool LoopCloser::DetectLoop(SLAM* system, Map* map, KeyFrame* kf) {
		auto db = map->mpKeyFrameDB;
		kf->SetNotErase();
		if (kf->mnId < map->mnLastLoopKFid + 10) {
			db->add(kf);
			kf->SetErase();
			return false;
		}
		auto vpConnectedKeyFrames = kf->GetVectorCovisibleKeyFrames();
		DBoW3::BowVector &CurrentBowVec = kf->mBowVec;
		float minScore = 1;
		for (size_t i = 0; i<vpConnectedKeyFrames.size(); i++)
		{
			KeyFrame* pKF = vpConnectedKeyFrames[i];
			if (pKF->isBad())
				continue;
			const DBoW3::BowVector &BowVec = pKF->mBowVec;

			float score = map->mpVoc->score(CurrentBowVec, BowVec);

			if (score<minScore)
				minScore = score;
		}
		auto  vpCandidateKFs = db->DetectLoopCandidates(kf, minScore);
		if (vpCandidateKFs.empty())
		{
			db->add(kf);
			map->mvConsistentGroups.clear();
			kf->SetErase();
			return false;
		}
		map->mvpEnoughConsistentCandidates.clear();
		
		std::vector<ConsistentGroup> vCurrentConsistentGroups;
		std::vector<bool> vbConsistentGroup(map->mvConsistentGroups.size(), false);
		for (size_t i = 0, iend = vpCandidateKFs.size(); i<iend; i++)
		{
			KeyFrame* pCandidateKF = vpCandidateKFs[i];

			auto spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
			spCandidateGroup.insert(pCandidateKF);

			bool bEnoughConsistent = false;
			bool bConsistentForSomeGroup = false;
			for (size_t iG = 0, iendG = map->mvConsistentGroups.size(); iG<iendG; iG++)
			{
				auto sPreviousGroup = map->mvConsistentGroups[iG].first;

				bool bConsistent = false;
				for (auto sit = spCandidateGroup.begin(), send = spCandidateGroup.end(); sit != send; sit++)
				{
					if (sPreviousGroup.count(*sit))
					{
						bConsistent = true;
						bConsistentForSomeGroup = true;
						break;
					}
				}

				if (bConsistent)
				{
					int nPreviousConsistency = map->mvConsistentGroups[iG].second;
					int nCurrentConsistency = nPreviousConsistency + 1;
					if (!vbConsistentGroup[iG])
					{
						ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
						vCurrentConsistentGroups.push_back(cg);
						vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
					}
					if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
					{
						map->mvpEnoughConsistentCandidates.push_back(pCandidateKF);
						bEnoughConsistent = true; //this avoid to insert the same candidate more than once
					}
				}
			}

			// If the group is not consistent with any previous group insert with consistency counter set to zero
			if (!bConsistentForSomeGroup)
			{
				ConsistentGroup cg = make_pair(spCandidateGroup, 0);
				vCurrentConsistentGroups.push_back(cg);
			}
		}
		db->add(kf);
		// Update Covisibility Consistent Groups
		map->mvConsistentGroups = vCurrentConsistentGroups;
		if (map->mvpEnoughConsistentCandidates.empty())
		{
			kf->SetErase();
			return false;
		}
		return true;

	}
	bool LoopCloser::ComputeSim3(SLAM* system, Map* map, KeyFrame* kf){
		const int nInitialCandidates = map->mvpEnoughConsistentCandidates.size();
				
		std::vector<Sim3Solver*> vpSim3Solvers;
		vpSim3Solvers.resize(nInitialCandidates);

		std::vector<std::vector<MapPoint*> > vvpMapPointMatches;
		vvpMapPointMatches.resize(nInitialCandidates);

		std::vector<bool> vbDiscarded;
		vbDiscarded.resize(nInitialCandidates);

		int nCandidates = 0; //candidates with enough matches

		for (int i = 0; i<nInitialCandidates; i++)
		{
			KeyFrame* pKF = map->mvpEnoughConsistentCandidates[i];

			// avoid that local mapping erase it while it is being processed in this thread
			pKF->SetNotErase();

			if (pKF->isBad())
			{
				vbDiscarded[i] = true;
				continue;
			}
			int nmatches = SearchPoints::SearchKeyFrameByBoW(pKF->matcher, kf, pKF, vvpMapPointMatches[i], pKF->matcher->min_descriptor_distance, 0.75);
			if (nmatches<20)
			{
				vbDiscarded[i] = true;
				continue;
			}
			else
			{
				Sim3Solver* pSolver = new Sim3Solver(kf, pKF, vvpMapPointMatches[i], map->mbFixScale);
				pSolver->SetRansacParameters(0.99, 20, 300);
				vpSim3Solvers[i] = pSolver;
			}

			nCandidates++;
		}
		bool bMatch = false;
		std::cout << "ComputeSim #candidates = " << nCandidates << std::endl;
		// Perform alternatively RANSAC iterations for each candidate
		// until one is succesful or all fail
		while (nCandidates>0 && !bMatch)
		{
			for (int i = 0; i<nInitialCandidates; i++)
			{
				if (vbDiscarded[i])
					continue;

				KeyFrame* pKF = map->mvpEnoughConsistentCandidates[i];

				// Perform 5 Ransac Iterations
				std::vector<bool> vbInliers;
				int nInliers;
				bool bNoMore;

				Sim3Solver* pSolver = vpSim3Solvers[i];
				cv::Mat Scm = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

				// If Ransac reachs max. iterations discard keyframe
				if (bNoMore)
				{
					vbDiscarded[i] = true;
					nCandidates--;
				}

				// If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
				if (!Scm.empty())
				{
					std::vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
					for (size_t j = 0, jend = vbInliers.size(); j<jend; j++)
					{
						if (vbInliers[j])
							vpMapPointMatches[j] = vvpMapPointMatches[i][j];
					}

					cv::Mat R = pSolver->GetEstimatedRotation();
					cv::Mat t = pSolver->GetEstimatedTranslation();
					const float s = pSolver->GetEstimatedScale();
					SearchPoints::SearchBySim3(kf->matcher, kf, pKF, vpMapPointMatches, s, R, t, kf->matcher->max_descriptor_distance);

					g2o::Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t), s);
					const int nInliers = Optimizer::OptimizeSim3(kf, pKF, vpMapPointMatches, gScm, 10, map->mbFixScale);

					// If optimization is succesful stop ransacs and continue
					if (nInliers >= 20)
					{
						bMatch = true;
						map->mpMatchedKF = pKF;
						g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()), Converter::toVector3d(pKF->GetTranslation()), 1.0);
						map->mg2oScw = gScm*gSmw;
						map->mScw = Converter::toCvMat(map->mg2oScw);
						map->mvpCurrentMatchedPoints = vpMapPointMatches;
						break;
					}
				}
			}
		}
		if (!bMatch)
		{
			for (int i = 0; i<nInitialCandidates; i++)
				map->mvpEnoughConsistentCandidates[i]->SetErase();
			kf->SetErase();
			std::cout << "false" << std::endl;
			return false;
		}

		// Retrieve MapPoints seen in Loop Keyframe and neighbors
		std::vector<KeyFrame*> vpLoopConnectedKFs = map->mpMatchedKF->GetVectorCovisibleKeyFrames();
		vpLoopConnectedKFs.push_back(map->mpMatchedKF);
		map->mvpLoopMapPoints.clear();
		for (auto vit = vpLoopConnectedKFs.begin(); vit != vpLoopConnectedKFs.end(); vit++)
		{
			KeyFrame* pKF = *vit;
			std::vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
			for (size_t i = 0, iend = vpMapPoints.size(); i<iend; i++)
			{
				MapPoint* pMP = vpMapPoints[i];
				if (pMP)
				{
					if (!pMP->isBad() && pMP->mnLoopPointForKF != kf->mnId)
					{
						map->mvpLoopMapPoints.push_back(pMP);
						pMP->mnLoopPointForKF = kf->mnId;
					}
				}
			}
		}
		std::cout << "3" << std::endl;
		// Find more matches projecting with the computed Sim3
		SearchPoints::SearchKeyByProjection(kf->matcher, kf, map->mScw, map->mvpLoopMapPoints, map->mvpCurrentMatchedPoints, kf->matcher->min_descriptor_distance);
		std::cout << "4" << std::endl;
		// If enough matches accept Loop
		int nTotalMatches = 0;
		for (size_t i = 0; i<map->mvpCurrentMatchedPoints.size(); i++)
		{
			if (map->mvpCurrentMatchedPoints[i])
				nTotalMatches++;
		}

		if (nTotalMatches >= 40)
		{
			for (int i = 0; i<nInitialCandidates; i++)
				if (map->mvpEnoughConsistentCandidates[i] != map->mpMatchedKF)
					map->mvpEnoughConsistentCandidates[i]->SetErase();
			return true;
		}
		for (int i = 0; i<nInitialCandidates; i++)
			map->mvpEnoughConsistentCandidates[i]->SetErase();
		kf->SetErase();
		return false;
	}

	void LoopCloser::CorrectLoop(SLAM* system, Map* map, KeyFrame* kf)
	{
		
		std::cout << "Loop detected!" << std::endl;

		// Send a stop signal to Local Mapping
		// Avoid new keyframes are inserted while correcting the loop
		map->RequestStop();

		// If a Global Bundle Adjustment is running, abort it
		if (map->isRunningGBA())
		{
			std::unique_lock<std::mutex> lock(map->mMutexGBA);
			map->mbStopGBA = true;

			map->mnFullBAIdx++;

			if (map->mpThreadGBA)
			{
				map->mpThreadGBA->detach();
				delete map->mpThreadGBA;
			}
		}

		// Wait until Local Mapping has effectively stopped
		while (!map->isStopped())
		{
			Sleep(1000);
		}

		// Ensure current keyframe is updated
		kf->UpdateConnections();

		// Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
		map->mvpCurrentConnectedKFs = kf->GetVectorCovisibleKeyFrames();
		map->mvpCurrentConnectedKFs.push_back(kf);

		KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
		CorrectedSim3[kf] = map->mg2oScw;
		cv::Mat Twc = kf->GetPoseInverse();


		{
			// Get Map Mutex
			std::unique_lock < std::mutex > lock(map->mMutexMapUpdate);

			for (auto vit = map->mvpCurrentConnectedKFs.begin(), vend = map->mvpCurrentConnectedKFs.end(); vit != vend; vit++)
			{
				KeyFrame* pKFi = *vit;

				cv::Mat Tiw = pKFi->GetPose();

				if (pKFi != kf)
				{
					cv::Mat Tic = Tiw*Twc;
					cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
					cv::Mat tic = Tic.rowRange(0, 3).col(3);
					g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
					g2o::Sim3 g2oCorrectedSiw = g2oSic*map->mg2oScw;
					//Pose corrected with the Sim3 of the loop closure
					CorrectedSim3[pKFi] = g2oCorrectedSiw;
				}

				cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
				cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
				g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
				//Pose without correction
				NonCorrectedSim3[pKFi] = g2oSiw;
			}

			// Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
			for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(), mend = CorrectedSim3.end(); mit != mend; mit++)
			{
				KeyFrame* pKFi = mit->first;
				g2o::Sim3 g2oCorrectedSiw = mit->second;
				g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

				g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

				std::vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
				for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
				{
					MapPoint* pMPi = vpMPsi[iMP];
					if (!pMPi)
						continue;
					if (pMPi->isBad())
						continue;
					if (pMPi->mnCorrectedByKF == kf->mnId)
						continue;

					// Project with non-corrected pose and project back with corrected pose
					cv::Mat P3Dw = pMPi->GetWorldPos();
					Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
					Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

					cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
					pMPi->SetWorldPos(cvCorrectedP3Dw);
					pMPi->mnCorrectedByKF = kf->mnId;
					pMPi->mnCorrectedReference = pKFi->mnId;
					pMPi->UpdateNormalAndDepth();
				}

				// Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
				Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
				Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
				double s = g2oCorrectedSiw.scale();

				eigt *= (1. / s); //[R t/s;0 1]

				cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);

				pKFi->SetPose(correctedTiw);

				// Make sure connections are updated
				pKFi->UpdateConnections();
			}

			// Start Loop Fusion
			// Update matched map points and replace if duplicated
			for (size_t i = 0; i<map->mvpCurrentMatchedPoints.size(); i++)
			{
				if (map->mvpCurrentMatchedPoints[i])
				{
					MapPoint* pLoopMP = map->mvpCurrentMatchedPoints[i];
					MapPoint* pCurMP = kf->GetMapPoint(i);
					if (pCurMP)
						pCurMP->Replace(pLoopMP);
					else
					{
						kf->AddMapPoint(pLoopMP, i);
						pLoopMP->AddObservation(kf, i);
						pLoopMP->ComputeDistinctiveDescriptors();
					}
				}
			}

		}

		// Project MapPoints observed in the neighborhood of the loop keyframe
		// into the current keyframe and neighbors using corrected poses.
		// Fuse duplications.
		SearchAndFuse(map, CorrectedSim3);


		// After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
		std::map<KeyFrame*, std::set<KeyFrame*> > LoopConnections;

		for (auto vit = map->mvpCurrentConnectedKFs.begin(), vend = map->mvpCurrentConnectedKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;
			std::vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

			// Update connections. Detect new links.
			pKFi->UpdateConnections();
			LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
			for (auto vit_prev = vpPreviousNeighbors.begin(), vend_prev = vpPreviousNeighbors.end(); vit_prev != vend_prev; vit_prev++)
			{
				LoopConnections[pKFi].erase(*vit_prev);
			}
			for (auto vit2 = map->mvpCurrentConnectedKFs.begin(), vend2 = map->mvpCurrentConnectedKFs.end(); vit2 != vend2; vit2++)
			{
				LoopConnections[pKFi].erase(*vit2);
			}
		}

		// Optimize graph
		Optimizer::OptimizeEssentialGraph(map, map->mpMatchedKF, kf, NonCorrectedSim3, CorrectedSim3, LoopConnections, map->mbFixScale);

		map->InformNewBigChange();

		// Add loop edge
		map->mpMatchedKF->AddLoopEdge(kf);
		kf->AddLoopEdge(map->mpMatchedKF);

		// Launch a new thread to perform Global Bundle Adjustment
		map->mbRunningGBA = true;
		map->mbFinishedGBA = false;
		map->mbStopGBA = false;
		map->mpThreadGBA = new std::thread(&LoopCloser::RunGlobalBundleAdjustment, system->mpLoopCloser, system, map, kf, kf->mnId);//

		// Loop closed. Release Local Mapping.
		map->Release();

		map->mnLastLoopKFid = kf->mnId;
	}

	void LoopCloser::SearchAndFuse(Map* map ,const KeyFrameAndPose &CorrectedPosesMap)
	{
		for (KeyFrameAndPose::const_iterator mit = CorrectedPosesMap.begin(), mend = CorrectedPosesMap.end(); mit != mend; mit++)
		{
			KeyFrame* pKF = mit->first;

			g2o::Sim3 g2oScw = mit->second;
			cv::Mat cvScw = Converter::toCvMat(g2oScw);

			std::vector<MapPoint*> vpReplacePoints(map->mvpLoopMapPoints.size(), static_cast<MapPoint*>(nullptr));
			SearchPoints::Fuse(pKF->matcher, pKF, cvScw, map->mvpLoopMapPoints, vpReplacePoints, pKF->matcher->min_descriptor_distance, 4.0);

			// Get Map Mutex
			std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);
			const int nLP = map->mvpLoopMapPoints.size();
			for (int i = 0; i<nLP; i++)
			{
				MapPoint* pRep = vpReplacePoints[i];
				if (pRep)
				{
					pRep->Replace(map->mvpLoopMapPoints[i]);
				}
			}
		}
	}
	void LoopCloser::RunGlobalBundleAdjustment(SLAM* system, Map* map, KeyFrame* kf, int nLoopKF)
	{
		std::cout << "Starting Global Bundle Adjustment" << std::endl;

		int idx = map->mnFullBAIdx;
		Optimizer::GlobalBundleAdjustemnt(map, 10, &map->mbStopGBA, nLoopKF, false);

		// Update all MapPoints and KeyFrames
		// Local Mapping was active during BA, that means that there might be new keyframes
		// not included in the Global BA and they are not consistent with the updated map.
		// We need to propagate the correction through the spanning tree
		{
			std::unique_lock<std::mutex> lock(map->mMutexGBA);
			if (idx != map->mnFullBAIdx)
				return;

			if (!map->mbStopGBA)
			{
				std::cout << "Global Bundle Adjustment finished" << std::endl;
				std::cout << "Updating map ..." << std::endl;
				map->RequestStop();
				// Wait until Local Mapping has effectively stopped

				while (!map->isStopped() && !map->isFinished())
				{
					Sleep(1000);
				}

				// Get Map Mutex
				std::unique_lock<std::mutex> lock(map->mMutexMapUpdate);

				// Correct keyframes starting at map first keyframe
				std::list<KeyFrame*> lpKFtoCheck(map->mvpKeyFrameOrigins.begin(), map->mvpKeyFrameOrigins.end());

				while (!lpKFtoCheck.empty())
				{
					KeyFrame* pKF = lpKFtoCheck.front();
					const std::set<KeyFrame*> sChilds = pKF->GetChilds();
					cv::Mat Twc = pKF->GetPoseInverse();
					for (std::set<KeyFrame*>::const_iterator sit = sChilds.begin(); sit != sChilds.end(); sit++)
					{
						KeyFrame* pChild = *sit;
						if (pChild->mnBAGlobalForKF != nLoopKF)
						{
							cv::Mat Tchildc = pChild->GetPose()*Twc;
							pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
							pChild->mnBAGlobalForKF = nLoopKF;

						}
						lpKFtoCheck.push_back(pChild);
					}

					pKF->mTcwBefGBA = pKF->GetPose();
					pKF->SetPose(pKF->mTcwGBA);
					lpKFtoCheck.pop_front();
				}

				// Correct MapPoints
				const std::vector<MapPoint*> vpMPs = map->GetAllMapPoints();

				for (size_t i = 0; i<vpMPs.size(); i++)
				{
					MapPoint* pMP = vpMPs[i];

					if (pMP->isBad())
						continue;

					if (pMP->mnBAGlobalForKF == nLoopKF)
					{
						// If optimized by Global BA, just update
						pMP->SetWorldPos(pMP->mPosGBA);
					}
					else
					{
						// Update according to the correction of its reference keyframe
						KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

						if (pRefKF->mnBAGlobalForKF != nLoopKF)
							continue;

						// Map to non-corrected camera
						cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
						cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
						cv::Mat Xc = Rcw*pMP->GetWorldPos() + tcw;

						// Backproject using corrected camera
						cv::Mat Twc = pRefKF->GetPoseInverse();
						cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
						cv::Mat twc = Twc.rowRange(0, 3).col(3);

						pMP->SetWorldPos(Rwc*Xc + twc);
					}
				}

				map->InformNewBigChange();
				map->Release();

				std::cout << "Map updated!" << std::endl;
			}

			map->mbFinishedGBA = true;
			map->mbRunningGBA = false;
		}
	}
}