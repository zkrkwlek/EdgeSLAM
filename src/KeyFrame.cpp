#include <KeyFrame.h>
#include <Frame.h>
#include <Map.h>
#include <MapPoint.h>
#include <Camera.h>
#include <FeatureTracker.h>
#include <CameraPose.h>
#include <Converter.h>

namespace EdgeSLAM {
	KeyFrame::KeyFrame(Frame *F, Map* pMap):
		mnId(pMap->mnNextKeyFrameID++),mnFrameId(F->mnFrameID), mdTimeStamp(F->mdTimeStamp), mnGridCols(F->FRAME_GRID_COLS), mnGridRows(F->FRAME_GRID_ROWS), mpCamPose(F->mpCamPose),
		mfGridElementWidthInv(F->mfGridElementWidthInv), mfGridElementHeightInv(F->mfGridElementHeightInv),
		mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),//mnTrackReferenceForFrame(0), 
		mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
		fx(F->fx), fy(F->fy), cx(F->cx), cy(F->cy), invfx(F->invfx), invfy(F->invfy),
		N(F->N), mvKeys(F->mvKeys), mvKeysUn(F->mvKeysUn),//mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), mvuRight(F.mvuRight), mvDepth(F.mvDepth),   //mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mpKeyFrameDB(pKFDB), mpORBvocabulary(F.mpORBvocabulary), mHalfBaseline(F.mb / 2), 
		mDescriptors(F->mDescriptors.clone()), mnScaleLevels(F->mnScaleLevels), mfScaleFactor(F->mfScaleFactor),
		mfLogScaleFactor(F->mfLogScaleFactor), mvScaleFactors(F->mvScaleFactors), mvLevelSigma2(F->mvLevelSigma2),
		mvInvLevelSigma2(F->mvInvLevelSigma2), mnMinX(F->mnMinX), mnMinY(F->mnMinY), mnMaxX(F->mnMaxX),
		mnMaxY(F->mnMaxY), K(F->K), mvpMapPoints(F->mvpMapPoints), 
		mbFirstConnection(true), mpParent(nullptr), mbNotErase(false),
		mbToBeErased(false), mbBad(false), mpMap(pMap), mpCamera(F->mpCamera)
	{
		F->mnKeyFrameId = this->mnId;
		mGrid.resize(mnGridCols);
		for (int i = 0; i<mnGridCols; i++)
		{
			mGrid[i].resize(mnGridRows);
			for (int j = 0; j<mnGridRows; j++)
				mGrid[i][j] = F->mGrid[i][j];
		}
	}
	KeyFrame:: ~KeyFrame(){}


	bool KeyFrame::is_in_image(float x, float y, float z){
		return mpCamera->is_in_image(x, y, z);
	}
	void KeyFrame::reset_map_points(){
		mvpMapPoints = std::vector<MapPoint*>(mvKeysUn.size(), nullptr);
		mvbOutliers = std::vector<bool>(mvKeysUn.size(), false);
	}

	////Covisibility
	void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
	{
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (!mConnectedKeyFrameWeights.count(pKF))
				mConnectedKeyFrameWeights[pKF] = weight;
			else if (mConnectedKeyFrameWeights[pKF] != weight)
				mConnectedKeyFrameWeights[pKF] = weight;
			else
				return;
		}

		UpdateBestCovisibles();
	}
	void KeyFrame::EraseConnection(KeyFrame* pKF)
	{
		bool bUpdate = false;
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (mConnectedKeyFrameWeights.count(pKF))
			{
				mConnectedKeyFrameWeights.erase(pKF);
				bUpdate = true;
			}
		}

		if (bUpdate)
			UpdateBestCovisibles();
	}
	void KeyFrame::UpdateConnections()
	{
		std::map<KeyFrame*, int> KFcounter;

		std::vector<MapPoint*> vpMP;

		{
			std::unique_lock<std::mutex> lockMPs(mMutexFeatures);
			vpMP = mvpMapPoints;
		}

		//For all map points in keyframe check in which other keyframes are they seen
		//Increase counter for those keyframes
		for (std::vector<MapPoint*>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
		{
			MapPoint* pMP = *vit;

			if (!pMP)
				continue;

			if (pMP->isBad())
				continue;

			std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

			for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				if (mit->first->mnId == mnId)
					continue;
				KFcounter[mit->first]++;
			}
		}

		// This should not happen
		if (KFcounter.empty())
			return;

		//If the counter is greater than threshold add connection
		//In case no keyframe counter is over threshold add the one with maximum counter
		int nmax = 0;
		KeyFrame* pKFmax = nullptr;
		int th = 15;

		std::vector<std::pair<int, KeyFrame*> > vPairs;
		vPairs.reserve(KFcounter.size());
		for (std::map<KeyFrame*, int>::iterator mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
		{
			if (mit->second>nmax)
			{
				nmax = mit->second;
				pKFmax = mit->first;
			}
			if (mit->second >= th)
			{
				vPairs.push_back(std::make_pair(mit->second, mit->first));
				(mit->first)->AddConnection(this, mit->second);
			}
		}

		if (vPairs.empty())
		{
			vPairs.push_back(std::make_pair(nmax, pKFmax));
			pKFmax->AddConnection(this, nmax);
		}

		sort(vPairs.begin(), vPairs.end());
		std::list<KeyFrame*> lKFs;
		std::list<int> lWs;
		for (size_t i = 0; i<vPairs.size(); i++)
		{
			lKFs.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		{
			std::unique_lock<std::mutex> lock(mMutexConnections);

			// mspConnectedKeyFrames = spConnectedKeyFrames;
			mConnectedKeyFrameWeights = KFcounter;
			mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
			mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

			if (mbFirstConnection && mnId != 0)
			{
				mpParent = mvpOrderedConnectedKeyFrames.front();
				mpParent->AddChild(this);
				mbFirstConnection = false;
			}

		}
	}
	void KeyFrame::UpdateBestCovisibles()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		std::vector<std::pair<int, KeyFrame*> > vPairs;
		vPairs.reserve(mConnectedKeyFrameWeights.size());
		for (std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
			vPairs.push_back(std::make_pair(mit->second, mit->first));

		sort(vPairs.begin(), vPairs.end());
		std::list<KeyFrame*> lKFs;
		std::list<int> lWs;
		for (size_t i = 0, iend = vPairs.size(); i<iend; i++)
		{
			lKFs.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		mvpOrderedConnectedKeyFrames = std::vector<KeyFrame*>(lKFs.begin(), lKFs.end());
		mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());
	}

	std::set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		std::set<KeyFrame*> s;
		for (std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(); mit != mConnectedKeyFrameWeights.end(); mit++)
			s.insert(mit->first);
		return s;
	}

	std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		return mvpOrderedConnectedKeyFrames;
	}

	std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		if ((int)mvpOrderedConnectedKeyFrames.size()<N)
			return mvpOrderedConnectedKeyFrames;
		else
			return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);

	}

	std::vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);

		if (mvpOrderedConnectedKeyFrames.empty())
			return std::vector<KeyFrame*>();

		std::vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, KeyFrame::weightComp);
		if (it == mvOrderedWeights.end())
			return std::vector<KeyFrame*>();
		else
		{
			int n = it - mvOrderedWeights.begin();
			return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
		}
	}

	int KeyFrame::GetWeight(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		if (mConnectedKeyFrameWeights.count(pKF))
			return mConnectedKeyFrameWeights[pKF];
		else
			return 0;
	}
	////Covisibility

	////Spanning Tree
	void KeyFrame::AddChild(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mspChildrens.insert(pKF);
	}

	void KeyFrame::EraseChild(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mspChildrens.erase(pKF);
	}

	void KeyFrame::ChangeParent(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mpParent = pKF;
		pKF->AddChild(this);
	}

	std::set<KeyFrame*> KeyFrame::GetChilds()
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mspChildrens;
	}

	KeyFrame* KeyFrame::GetParent()
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mpParent;
	}

	bool KeyFrame::hasChild(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mspChildrens.count(pKF);
	}
	
	////Spanning Tree

	////Loop Edges
	void KeyFrame::AddLoopEdge(KeyFrame *pKF)
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		mbNotErase = true;
		mspLoopEdges.insert(pKF);
	}

	std::set<KeyFrame*> KeyFrame::GetLoopEdges()
	{
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mspLoopEdges;
	}
	////Loop Edges

	////Flag
	void KeyFrame::SetNotErase()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		mbNotErase = true;
	}

	void KeyFrame::SetErase()
	{
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (mspLoopEdges.empty())
			{
				mbNotErase = false;
			}
		}

		if (mbToBeErased)
		{
			SetBadFlag();
		}
	}

	void KeyFrame::SetBadFlag()
	{
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (mnId == 0)
				return;
			else if (mbNotErase)
			{
				mbToBeErased = true;
				return;
			}
		}
		
		std::vector<MapPoint*> vpMP;
		{
			std::unique_lock<std::mutex> lockMPs(mMutexFeatures);
			vpMP = mvpMapPoints;
		}

		for (std::map<KeyFrame*, int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend = mConnectedKeyFrameWeights.end(); mit != mend; mit++)
			mit->first->EraseConnection(this);

		for (size_t i = 0; i<vpMP.size(); i++)
			if (vpMP[i])
				vpMP[i]->EraseObservation(this);
		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			std::unique_lock<std::mutex> lock1(mMutexFeatures);

			mConnectedKeyFrameWeights.clear();
			mvpOrderedConnectedKeyFrames.clear();

			// Update Spanning Tree
			std::set<KeyFrame*> sParentCandidates;
			sParentCandidates.insert(mpParent);

			// Assign at each iteration one children with a parent (the pair with highest covisibility weight)
			// Include that children as new parent candidate for the rest
			while (!mspChildrens.empty())
			{
				bool bContinue = false;

				int max = -1;
				KeyFrame* pC = nullptr;
				KeyFrame* pP = nullptr;

				for (std::set<KeyFrame*>::iterator sit = mspChildrens.begin(), send = mspChildrens.end(); sit != send; sit++)
				{
					KeyFrame* pKF = *sit;
					if (pKF->isBad())
						continue;

					// Check if a parent candidate is connected to the keyframe
					std::vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
					for (size_t i = 0, iend = vpConnected.size(); i<iend; i++)
					{
						for (std::set<KeyFrame*>::iterator spcit = sParentCandidates.begin(), spcend = sParentCandidates.end(); spcit != spcend; spcit++)
						{
							if (vpConnected[i]->mnId == (*spcit)->mnId)
							{
								int w = pKF->GetWeight(vpConnected[i]);
								if (w>max)
								{
									pC = pKF;
									pP = vpConnected[i];
									max = w;
									bContinue = true;
								}
							}
						}
					}
				}

				if (bContinue && pC && pP)
				{
					pC->ChangeParent(pP);
					sParentCandidates.insert(pC);
					mspChildrens.erase(pC);
				}
				else
					break;
			}

			// If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
			if (!mspChildrens.empty())
				for (std::set<KeyFrame*>::iterator sit = mspChildrens.begin(); sit != mspChildrens.end(); sit++)
				{
					(*sit)->ChangeParent(mpParent);
				}

			mpParent->EraseChild(this);
			mTcp = mpCamPose->GetPose()*mpParent->GetPoseInverse();
			mbBad = true;
		}


		mpMap->RemoveKeyFrame(this);
		//mpKeyFrameDB->erase(this);
	}

	bool KeyFrame::isBad()
	{
		std::unique_lock<std::mutex> lock(mMutexConnections);
		return mbBad;
	}
	////Flag

	////Map Point
	void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		mvpMapPoints[idx] = pMP;
	}

	void KeyFrame::EraseMapPointMatch(const size_t &idx)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		mvpMapPoints[idx] = static_cast<MapPoint*>(nullptr);
	}

	void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		int idx = pMP->GetIndexInKeyFrame(this);
		if (idx >= 0)
			mvpMapPoints[idx] = static_cast<MapPoint*>(nullptr);
	}


	void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		mvpMapPoints[idx] = pMP;
	}

	std::set<MapPoint*> KeyFrame::GetMapPoints()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		std::set<MapPoint*> s;
		for (size_t i = 0, iend = mvpMapPoints.size(); i<iend; i++)
		{
			if (!mvpMapPoints[i])
				continue;
			MapPoint* pMP = mvpMapPoints[i];
			if (!pMP->isBad())
				s.insert(pMP);
		}
		return s;
	}

	int KeyFrame::TrackedMapPoints(const int &minObs)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);

		int nPoints = 0;
		const bool bCheckObs = minObs>0;
		for (int i = 0; i<N; i++)
		{
			MapPoint* pMP = mvpMapPoints[i];
			if (pMP)
			{
				if (!pMP->isBad())
				{
					if (bCheckObs)
					{
						if (mvpMapPoints[i]->Observations() >= minObs)
							nPoints++;
					}
					else
						nPoints++;
				}
			}
		}

		return nPoints;
	}

	std::vector<MapPoint*> KeyFrame::GetMapPointMatches()
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mvpMapPoints;
	}

	MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
	{
		std::unique_lock<std::mutex> lock(mMutexFeatures);
		return mvpMapPoints[idx];
	}
	std::vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
	{
		std::vector<size_t> vIndices;
		vIndices.reserve(N);

		const int nMinCellX = std::max(0, (int)floor((x - mnMinX - r)*mfGridElementWidthInv));
		if (nMinCellX >= mnGridCols)
			return vIndices;

		const int nMaxCellX = std::min((int)mnGridCols - 1, (int)ceil((x - mnMinX + r)*mfGridElementWidthInv));
		if (nMaxCellX<0)
			return vIndices;

		const int nMinCellY = std::max(0, (int)floor((y - mnMinY - r)*mfGridElementHeightInv));
		if (nMinCellY >= mnGridRows)
			return vIndices;

		const int nMaxCellY = std::min((int)mnGridRows - 1, (int)ceil((y - mnMinY + r)*mfGridElementHeightInv));
		if (nMaxCellY<0)
			return vIndices;

		for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
		{
			for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
			{
				const std::vector<size_t> vCell = mGrid[ix][iy];
				for (size_t j = 0, jend = vCell.size(); j<jend; j++)
				{
					const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
					const float distx = kpUn.pt.x - x;
					const float disty = kpUn.pt.y - y;

					if (fabs(distx)<r && fabs(disty)<r)
						vIndices.push_back(vCell[j]);
				}
			}
		}

		return vIndices;
	}
	float KeyFrame::ComputeSceneMedianDepth(const int q)
	{
		std::vector<MapPoint*> vpMapPoints;
		cv::Mat Tcw_;
		{
			std::unique_lock<std::mutex> lock(mMutexFeatures);
			std::unique_lock<std::mutex> lock2(mMutexPose);
			vpMapPoints = mvpMapPoints;
			Tcw_ = mpCamPose->GetPose();
		}

		std::vector<float> vDepths;
		vDepths.reserve(N);
		cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
		Rcw2 = Rcw2.t();
		float zcw = Tcw_.at<float>(2, 3);
		for (int i = 0; i<N; i++)
		{
			if (vpMapPoints[i])
			{
				MapPoint* pMP = vpMapPoints[i];
				cv::Mat x3Dw = pMP->GetWorldPos();
				float z = Rcw2.dot(x3Dw) + zcw;
				vDepths.push_back(z);
			}
		}

		sort(vDepths.begin(), vDepths.end());

		return vDepths[(vDepths.size() - 1) / q];
	}
	////Map Point

	////DBoW
	void KeyFrame::ComputeBoW()
	{
		if (mBowVec.empty())
		{
			std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
			mpVoc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);  // 5 is better
		}
	}
	////DBoW

	////Camera
	void KeyFrame::SetPose(const cv::Mat &Tcw) {
		mpCamPose->SetPose(Tcw);
	}
	cv::Mat KeyFrame::GetPose() {
		return mpCamPose->GetPose();
	}
	cv::Mat KeyFrame::GetPoseInverse() {
		return mpCamPose->GetInversePose();
	}
	cv::Mat KeyFrame::GetCameraCenter() {
		return mpCamPose->GetCenter();
	}
	cv::Mat KeyFrame::GetRotation() {
		return mpCamPose->GetRotation();
	}
	cv::Mat KeyFrame::GetTranslation() {
		return mpCamPose->GetTranslation();
	}
	////Camera
}
