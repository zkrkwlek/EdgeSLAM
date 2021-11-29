#include <SearchPoints.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <FeatureTracker.h>

namespace EdgeSLAM {
	const int SearchPoints::HISTO_LENGTH = 30;
	int SearchPoints::SearchObject(cv::Mat obj, DBoW3::FeatureVector fvec, Frame* curr, std::vector<std::pair<int, int>>& matches, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri) {
		int nmatches = 0;

		// Rotation Histogram (to check rotation consistency)
		std::vector<int> rotHist[HISTO_LENGTH];
		for (int i = 0; i<HISTO_LENGTH; i++)
			rotHist[i].reserve(500);
		const float factor = 1.0f / HISTO_LENGTH;


		std::vector<bool> bMatches = std::vector<bool>(curr->N, false);
		DBoW3::FeatureVector::const_iterator KFit = fvec.begin();
		DBoW3::FeatureVector::const_iterator Fit = curr->mFeatVec.begin();
		DBoW3::FeatureVector::const_iterator KFend = fvec.end();
		DBoW3::FeatureVector::const_iterator Fend = curr->mFeatVec.end();

		FeatureTracker* mpFeatureTracker = new FlannFeatureTracker(1500);

		while (KFit != KFend && Fit != Fend)
		{
			if (KFit->first == Fit->first)
			{
				const std::vector<unsigned int> vIndicesKF = KFit->second;
				const std::vector<unsigned int> vIndicesF = Fit->second;

				for (size_t iKF = 0; iKF<vIndicesKF.size(); iKF++)
				{
					const unsigned int realIdxKF = vIndicesKF[iKF];
					
					const cv::Mat &dKF = obj.row(realIdxKF);

					int bestDist1 = 256;
					int bestIdxF = -1;
					int bestDist2 = 256;

					for (size_t iF = 0; iF<vIndicesF.size(); iF++)
					{
						const unsigned int realIdxF = vIndicesF[iF];

						if (bMatches[realIdxF])
							continue;

						const cv::Mat &dF = curr->mDescriptors.row(realIdxF);

						const int dist = (int)mpFeatureTracker->DescriptorDistance(dKF, dF);
						std::cout <<"a "<< dist << std::endl;
						if (dist<bestDist1)
						{
							bestDist2 = bestDist1;
							bestDist1 = dist;
							bestIdxF = realIdxF;
						}
						else if (dist<bestDist2)
						{
							bestDist2 = dist;
						}
					}

					if (bestDist1 <= thMinDesc)
					{
						if (static_cast<float>(bestDist1)<thProjection*static_cast<float>(bestDist2))
						{
							bMatches[bestIdxF] = true;
							matches.push_back(std::make_pair((int)realIdxKF, (int)bestIdxF));
							nmatches++;
						}
					}

				}

				KFit++;
				Fit++;
			}
			else if (KFit->first < Fit->first)
			{
				KFit = fvec.lower_bound(Fit->first);
			}
			else
			{
				Fit = curr->mFeatVec.lower_bound(KFit->first);
			}
		}
		std::cout << "obj match = " << nmatches << std::endl;

		/*for (int i = 0; i < obj.rows; i++)
		{

			int bestDist = 256;
			int bestIdx2 = -1;
			const cv::Mat &d1 = obj.row(i);
			for (size_t j = 0, jend = curr->N; j < jend; j++) {
				const cv::Mat &d2 = curr->mDescriptors.row(j);

				const int dist = (int)curr->matcher->DescriptorDistance(d1, d2);

				if (dist < bestDist)
				{
					bestDist = dist;
					bestIdx2 = j;
				}
			}
			if (bestDist <= thMaxDesc)
			{
				nmatches++;
				matches.push_back(std::make_pair(i, bestIdx2));
			}
		}*/
		return nmatches;
	}

	int SearchPoints::SearchKeyFrameByBoW(KeyFrame* pKF1, KeyFrame *pKF2, std::vector<MapPoint*> &vpMatches12, float thMatchRatio, bool bCheckOri)
	{
		FeatureTracker* mpFeatureTracker = new FlannFeatureTracker(1500);
		const std::vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
		const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
		const std::vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
		const cv::Mat &Descriptors1 = pKF1->mDescriptors;

		const std::vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
		const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
		const std::vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
		const cv::Mat &Descriptors2 = pKF2->mDescriptors;

		vpMatches12 =std::vector<MapPoint*>(vpMapPoints1.size(), static_cast<MapPoint*>(NULL));
		std::vector<bool> vbMatched2(vpMapPoints2.size(), false);

		std::vector<int> rotHist[HISTO_LENGTH];
		for (int i = 0; i<HISTO_LENGTH; i++)
			rotHist[i].reserve(500);

		const float factor = 1.0f / HISTO_LENGTH;

		int nmatches = 0;

		DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
		DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
		DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
		DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

		while (f1it != f1end && f2it != f2end)
		{
			if (f1it->first == f2it->first)
			{
				for (size_t i1 = 0, iend1 = f1it->second.size(); i1<iend1; i1++)
				{
					const size_t idx1 = f1it->second[i1];

					MapPoint* pMP1 = vpMapPoints1[idx1];
					if (!pMP1)
						continue;
					if (pMP1->isBad())
						continue;

					const cv::Mat &d1 = Descriptors1.row(idx1);

					int bestDist1 = 256;
					int bestIdx2 = -1;
					int bestDist2 = 256;

					for (size_t i2 = 0, iend2 = f2it->second.size(); i2<iend2; i2++)
					{
						const size_t idx2 = f2it->second[i2];

						MapPoint* pMP2 = vpMapPoints2[idx2];

						if (vbMatched2[idx2] || !pMP2)
							continue;

						if (pMP2->isBad())
							continue;

						const cv::Mat &d2 = Descriptors2.row(idx2);

						int dist = mpFeatureTracker->DescriptorDistance(d1, d2);

						if (dist<bestDist1)
						{
							bestDist2 = bestDist1;
							bestDist1 = dist;
							bestIdx2 = idx2;
						}
						else if (dist<bestDist2)
						{
							bestDist2 = dist;
						}
					}

					if (bestDist1<mpFeatureTracker->min_descriptor_distance)
					{
						if (static_cast<float>(bestDist1)<thMatchRatio*static_cast<float>(bestDist2))
						{
							vpMatches12[idx1] = vpMapPoints2[bestIdx2];
							vbMatched2[bestIdx2] = true;

							if (bCheckOri)
							{
								float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
								if (rot<0.0)
									rot += 360.0f;
								int bin = round(rot*factor);
								if (bin == HISTO_LENGTH)
									bin = 0;
								assert(bin >= 0 && bin<HISTO_LENGTH);
								rotHist[bin].push_back(idx1);
							}
							nmatches++;
						}
					}
				}

				f1it++;
				f2it++;
			}
			else if (f1it->first < f2it->first)
			{
				f1it = vFeatVec1.lower_bound(f2it->first);
			}
			else
			{
				f2it = vFeatVec2.lower_bound(f1it->first);
			}
		}

		if (bCheckOri)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i<HISTO_LENGTH; i++)
			{
				if (i == ind1 || i == ind2 || i == ind3)
					continue;
				for (size_t j = 0, jend = rotHist[i].size(); j<jend; j++)
				{
					vpMatches12[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
					nmatches--;
				}
			}
		}

		return nmatches;
	}
	int SearchPoints::SearchFrameByBoW(KeyFrame* pKF, Frame *F, std::vector<MapPoint*> &vpMapPointMatches, float thMinDesc, float thMatchRatio, bool bCheckOri) {
		FeatureTracker* mpFeatureTracker = new FlannFeatureTracker(1500);
		const auto vpMapPointsKF = pKF->GetMapPointMatches();
		
		vpMapPointMatches = std::vector<MapPoint*>(F->N, static_cast<MapPoint*>(nullptr));

		const DBoW3::FeatureVector &vFeatVecKF = pKF->mFeatVec;
		int nmatches = 0;

		std::vector<int> rotHist[HISTO_LENGTH];
		for (int i = 0; i<HISTO_LENGTH; i++)
			rotHist[i].reserve(500);
		const float factor = 1.0f / HISTO_LENGTH;

		// We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
		DBoW3::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
		DBoW3::FeatureVector::const_iterator Fit = F->mFeatVec.begin();
		DBoW3::FeatureVector::const_iterator KFend = vFeatVecKF.end();
		DBoW3::FeatureVector::const_iterator Fend = F->mFeatVec.end();

		while (KFit != KFend && Fit != Fend)
		{
			if (KFit->first == Fit->first)
			{
				const std::vector<unsigned int> vIndicesKF = KFit->second;
				const std::vector<unsigned int> vIndicesF = Fit->second;

				for (size_t iKF = 0; iKF<vIndicesKF.size(); iKF++)
				{
					const unsigned int realIdxKF = vIndicesKF[iKF];

					MapPoint* pMP = vpMapPointsKF[realIdxKF];

					if (!pMP || pMP->isBad())
						continue;

					const cv::Mat &dKF = pKF->mDescriptors.row(realIdxKF);

					int bestDist1 = 256;
					int bestIdxF = -1;
					int bestDist2 = 256;

					for (size_t iF = 0; iF<vIndicesF.size(); iF++)
					{
						const unsigned int realIdxF = vIndicesF[iF];

						if (vpMapPointMatches[realIdxF])
							continue;

						const cv::Mat &dF = F->mDescriptors.row(realIdxF);

						const int dist = (int)mpFeatureTracker->DescriptorDistance(dKF, dF);

						if (dist<bestDist1)
						{
							bestDist2 = bestDist1;
							bestDist1 = dist;
							bestIdxF = realIdxF;
						}
						else if (dist<bestDist2)
						{
							bestDist2 = dist;
						}
					}
					
					if (bestDist1 <= thMinDesc)
					{
						if (static_cast<float>(bestDist1)<thMatchRatio*static_cast<float>(bestDist2))
						{
							vpMapPointMatches[bestIdxF] = pMP;

							const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

							if (bCheckOri)
							{
								float rot = kp.angle - F->mvKeys[bestIdxF].angle;
								if (rot<0.0)
									rot += 360.0f;
								int bin = round(rot*factor);
								if (bin == HISTO_LENGTH)
									bin = 0;
								assert(bin >= 0 && bin<HISTO_LENGTH);
								rotHist[bin].push_back(bestIdxF);
							}
							nmatches++;
						}
					}

				}

				KFit++;
				Fit++;
			}
			else if (KFit->first < Fit->first)
			{
				KFit = vFeatVecKF.lower_bound(Fit->first);
			}
			else
			{
				Fit = F->mFeatVec.lower_bound(KFit->first);
			}
		}
		if (bCheckOri)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i<HISTO_LENGTH; i++)
			{
				if (i == ind1 || i == ind2 || i == ind3)
					continue;
				for (size_t j = 0, jend = rotHist[i].size(); j<jend; j++)
				{
					vpMapPointMatches[rotHist[i][j]] = static_cast<MapPoint*>(nullptr);
					nmatches--;
				}
			}
		}
		return nmatches;
	}

	int SearchPoints::SearchFrameByProjection(Frame* prev, Frame* curr, float thMaxDesc, float thMinDesc, float thProjection, bool bCheckOri) {

		int nmatches = 0;
		FeatureTracker* mpFeatureTracker = new FlannFeatureTracker(1500);

		// Rotation Histogram (to check rotation consistency)
		std::vector<int> rotHist[HISTO_LENGTH];
		for (int i = 0; i<HISTO_LENGTH; i++)
			rotHist[i].reserve(500);
		float factor = 1.0f / HISTO_LENGTH;
		cv::Mat Tcw = curr->GetPose();
		cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
		cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
		cv::Mat twc = -Rcw.t()*tcw;

		cv::Mat Tlw = prev->GetPose();
		cv::Mat Rlw = Tlw.rowRange(0, 3).colRange(0, 3);
		cv::Mat tlw = Tlw.rowRange(0, 3).col(3);
		cv::Mat tlc = Rlw*twc + tlw;

		for (int i = 0; i<prev->N; i++)
		{
			MapPoint* pMP = prev->mvpMapPoints[i];

			if (pMP)
			{
				if (!prev->mvbOutliers[i])
				{
					// Project
					cv::Mat x3Dw = pMP->GetWorldPos();
					cv::Mat x3Dc = Rcw*x3Dw + tcw;

					float xc = x3Dc.at<float>(0);
					float yc = x3Dc.at<float>(1);
					float invzc = 1.0 / x3Dc.at<float>(2);

					if (invzc<0)
						continue;

					float u = curr->fx*xc*invzc + curr->cx;
					float v = curr->fy*yc*invzc + curr->cy;

					if (u<curr->mnMinX || u>curr->mnMaxX)
						continue;
					if (v<curr->mnMinY || v>curr->mnMaxY)
						continue;

					int nLastOctave = prev->mvKeys[i].octave;

					// Search in a window. Size depends on scale
					float radius = thProjection*curr->mvScaleFactors[nLastOctave];

					std::vector<size_t> vIndices2;
					vIndices2 = curr->GetFeaturesInArea(u, v, radius, nLastOctave - 1, nLastOctave + 1);
					if (vIndices2.empty())
						continue;

					cv::Mat dMP = pMP->GetDescriptor();

					int bestDist = 256;
					int bestIdx2 = -1;

					for (std::vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
					{
						size_t i2 = *vit;
						if (curr->mvpMapPoints[i2])
							if (curr->mvpMapPoints[i2]->Observations()>0)
								continue;

						cv::Mat &d = curr->mDescriptors.row(i2);

						int dist = (int)mpFeatureTracker->DescriptorDistance(dMP, d);

						if (dist<bestDist)
						{
							bestDist = dist;
							bestIdx2 = i2;
						}
					}

					if (bestDist <= thMaxDesc)
					{
						curr->mvpMapPoints[bestIdx2] = pMP;
						nmatches++;

						if (bCheckOri)
						{
							float rot = prev->mvKeysUn[i].angle - curr->mvKeysUn[bestIdx2].angle;
							if (rot<0.0)
								rot += 360.0f;
							int bin = round(rot*factor);
							if (bin == HISTO_LENGTH)
								bin = 0;
							assert(bin >= 0 && bin<HISTO_LENGTH);
							rotHist[bin].push_back(bestIdx2);
						}
					}
				}
			}
		}

		if (bCheckOri)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i<HISTO_LENGTH; i++)
			{
				if (i != ind1 && i != ind2 && i != ind3)
				{
					for (size_t j = 0, jend = rotHist[i].size(); j<jend; j++)
					{
						curr->mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint*>(nullptr);
						nmatches--;
					}
				}
			}
		}
		return nmatches;
	}

	int SearchPoints::SearchFrameByProjection(Frame *pF, KeyFrame *pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist, bool bCheckOri) {
		FeatureTracker* mpFeatureTracker = new FlannFeatureTracker(1500);
		int nmatches = 0;


		const cv::Mat Rcw = pF->GetRotation();
		const cv::Mat tcw = pF->GetTranslation();
		const cv::Mat Ow = pF->GetCameraCenter();

		// Rotation Histogram (to check rotation consistency)
		std::vector<int> rotHist[HISTO_LENGTH];
		for (int i = 0; i<HISTO_LENGTH; i++)
			rotHist[i].reserve(500);
		const float factor = 1.0f / HISTO_LENGTH;

		const auto vpMPs = pKF->GetMapPointMatches();

		float fx = pF->fx;
		float fy = pF->fy;
		float cx = pF->cx;
		float cy = pF->cy;
		float nMinX = pF->mnMinX;
		float nMinY = pF->mnMinY;
		float nMaxX = pF->mnMaxX;
		float nMaxY = pF->mnMaxY;

		for (size_t i = 0, iend = vpMPs.size(); i<iend; i++)
		{
			MapPoint* pMP = vpMPs[i];

			if (pMP)
			{
				if (!pMP->isBad() && !sAlreadyFound.count(pMP))
				{
					//Project
					cv::Mat x3Dw = pMP->GetWorldPos();
					cv::Mat x3Dc = Rcw*x3Dw + tcw;

					const float xc = x3Dc.at<float>(0);
					const float yc = x3Dc.at<float>(1);
					const float invzc = 1.0 / x3Dc.at<float>(2);

					const float u = fx*xc*invzc + cx;
					const float v = fy*yc*invzc + cy;

					if (u<nMinX || u>nMaxX)
						continue;
					if (v<nMinY || v>nMaxY)
						continue;

					// Compute predicted scale level
					cv::Mat PO = x3Dw - Ow;
					float dist3D = cv::norm(PO);

					const float maxDistance = pMP->GetMaxDistanceInvariance();
					const float minDistance = pMP->GetMinDistanceInvariance();

					// Depth must be inside the scale pyramid of the image
					if (dist3D<minDistance || dist3D>maxDistance)
						continue;

					int nPredictedLevel = pMP->PredictScale(dist3D, pF);

					// Search in a window
					const float radius = th*pF->mvScaleFactors[nPredictedLevel];

					const auto vIndices2 = pF->GetFeaturesInArea(u, v, radius, nPredictedLevel - 1, nPredictedLevel + 1);

					if (vIndices2.empty())
						continue;

					const cv::Mat dMP = pMP->GetDescriptor();

					int bestDist = 256;
					int bestIdx2 = -1;

					for (auto vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
					{
						const size_t i2 = *vit;
						if (pF->mvpMapPoints[i2])
							continue;

						const cv::Mat &d = pF->mDescriptors.row(i2);

						const int dist = (int)mpFeatureTracker->DescriptorDistance(dMP, d);

						if (dist<bestDist)
						{
							bestDist = dist;
							bestIdx2 = i2;
						}
					}

					if (bestDist <= ORBdist)
					{
						pF->mvpMapPoints[bestIdx2] = pMP;
						nmatches++;

						if (bCheckOri)
						{
							float rot = pKF->mvKeysUn[i].angle - pF->mvKeysUn[bestIdx2].angle;
							if (rot<0.0)
								rot += 360.0f;
							int bin = round(rot*factor);
							if (bin == HISTO_LENGTH)
								bin = 0;
							assert(bin >= 0 && bin<HISTO_LENGTH);
							rotHist[bin].push_back(bestIdx2);
						}
					}

				}
			}
		}

		if (bCheckOri)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i<HISTO_LENGTH; i++)
			{
				if (i != ind1 && i != ind2 && i != ind3)
				{
					for (size_t j = 0, jend = rotHist[i].size(); j<jend; j++)
					{
						pF->mvpMapPoints[rotHist[i][j]] = nullptr;
						nmatches--;
					}
				}
			}
		}

		return nmatches;
	}
	int SearchPoints::SearchKeyByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, float thRadius) {
		FeatureTracker* mpFeatureTracker = new FlannFeatureTracker(1500);
		// Get Calibration Parameters for later projection
		const float &fx = pKF->fx;
		const float &fy = pKF->fy;
		const float &cx = pKF->cx;
		const float &cy = pKF->cy;

		// Decompose Scw
		cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
		const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
		cv::Mat Rcw = sRcw / scw;
		cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
		cv::Mat Ow = -Rcw.t()*tcw;

		// Set of MapPoints already found in the KeyFrame
		std::set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
		spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

		int nmatches = 0;

		// For each Candidate MapPoint Project and Match
		for (int iMP = 0, iendMP = vpPoints.size(); iMP<iendMP; iMP++)
		{
			MapPoint* pMP = vpPoints[iMP];

			// Discard Bad MapPoints and already found
			if (pMP->isBad() || spAlreadyFound.count(pMP))
				continue;

			// Get 3D Coords.
			cv::Mat p3Dw = pMP->GetWorldPos();

			// Transform into Camera Coords.
			cv::Mat p3Dc = Rcw*p3Dw + tcw;

			// Depth must be positive
			if (p3Dc.at<float>(2)<0.0)
				continue;

			// Project into Image
			const float invz = 1 / p3Dc.at<float>(2);
			const float x = p3Dc.at<float>(0)*invz;
			const float y = p3Dc.at<float>(1)*invz;

			const float u = fx*x + cx;
			const float v = fy*y + cy;

			// Point must be inside the image
			if (!pKF->is_in_image(u, v))
				continue;

			// Depth must be inside the scale invariance region of the point
			const float maxDistance = pMP->GetMaxDistanceInvariance();
			const float minDistance = pMP->GetMinDistanceInvariance();
			cv::Mat PO = p3Dw - Ow;
			const float dist = cv::norm(PO);

			if (dist<minDistance || dist>maxDistance)
				continue;

			// Viewing angle must be less than 60 deg
			cv::Mat Pn = pMP->GetNormal();

			if (PO.dot(Pn)<0.5*dist)
				continue;

			int nPredictedLevel = pMP->PredictScale(dist, pKF);

			// Search in a radius
			const float radius = thRadius*pKF->mvScaleFactors[nPredictedLevel];

			const std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

			if (vIndices.empty())
				continue;

			// Match to the most similar keypoint in the radius
			const cv::Mat dMP = pMP->GetDescriptor();

			int bestDist = 256;
			int bestIdx = -1;
			for (auto vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
			{
				const size_t idx = *vit;
				if (vpMatched[idx])
					continue;

				const int &kpLevel = pKF->mvKeysUn[idx].octave;

				if (kpLevel<nPredictedLevel - 1 || kpLevel>nPredictedLevel)
					continue;

				const cv::Mat &dKF = pKF->mDescriptors.row(idx);

				const int dist = mpFeatureTracker->DescriptorDistance(dMP, dKF);

				if (dist<bestDist)
				{
					bestDist = dist;
					bestIdx = idx;
				}
			}

			if (bestDist <= mpFeatureTracker->min_descriptor_distance)
			{
				vpMatched[bestIdx] = pMP;
				nmatches++;
			}

		}

		return nmatches;
	}
	int SearchPoints::SearchMapByProjection(Frame *F, const std::vector<MapPoint*> &vpMapPoints, const std::vector<TrackPoint*> &vpTrackPoints, float thMaxDesc, float thMinDesc, float thRadius, float thMatchRatio, bool bCheckOri)
	{
		FeatureTracker* mpFeatureTracker = new FlannFeatureTracker(1500);
		int nmatches = 0;
		const bool bFactor = thRadius != 1.0;

		for (size_t iMP = 0; iMP<vpMapPoints.size(); iMP++)
		{
			MapPoint* pMP = vpMapPoints[iMP];
			TrackPoint* pTP = vpTrackPoints[iMP];

			if (!pTP->mbTrackInView || pMP->isBad()){
				continue;
			}
			const int &nPredictedLevel = pTP->mnTrackScaleLevel;

			// The size of the window will depend on the viewing direction
			float r = RadiusByViewingCos(pTP->mTrackViewCos);

			if (bFactor)
				r *= thRadius;

			const std::vector<size_t> vIndices =
				F->GetFeaturesInArea(pTP->mTrackProjX, pTP->mTrackProjY, r*F->mvScaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel);
			
			if (vIndices.empty()){
				continue;
			}
			const cv::Mat MPdescriptor = pMP->GetDescriptor();

			int bestDist = 256;
			int bestLevel = -1;
			int bestDist2 = 256;
			int bestLevel2 = -1;
			int bestIdx = -1;

			// Get best and second matches with near keypoints
			for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
			{
				const size_t idx = *vit;

				if (F->mvpMapPoints[idx] && F->mvpMapPoints[idx]->Observations()>0){
					continue;
				}
				const cv::Mat &d = F->mDescriptors.row(idx);

				const int dist = (int)mpFeatureTracker->DescriptorDistance(MPdescriptor, d);

				if (dist<bestDist)
				{
					bestDist2 = bestDist;
					bestDist = dist;
					bestLevel2 = bestLevel;
					bestLevel = F->mvKeysUn[idx].octave;
					bestIdx = idx;
				}
				else if (dist<bestDist2)
				{
					bestLevel2 = F->mvKeysUn[idx].octave;
					bestDist2 = dist;
				}
			}
			// Apply ratio to second match (only if best and second are in the same scale level)
			if (bestDist <= thMaxDesc)
			{
				if (bestLevel == bestLevel2 && bestDist > thMatchRatio*bestDist2){
					continue;
				}
				F->mvpMapPoints[bestIdx] = pMP;
				F->mvpTrackPoints[bestIdx] = pTP;
				nmatches++;
			}
		}

		return nmatches;
	}

	int SearchPoints::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, std::vector<MapPoint*> &vpMatches12,
		const float &s12, const cv::Mat &R12, const cv::Mat &t12, float thRadius)
	{
		FeatureTracker* mpFeatureTracker = new FlannFeatureTracker(1500);
		const float &fx = pKF1->fx;
		const float &fy = pKF1->fy;
		const float &cx = pKF1->cx;
		const float &cy = pKF1->cy;

		// Camera 1 from world
		cv::Mat R1w = pKF1->GetRotation();
		cv::Mat t1w = pKF1->GetTranslation();

		//Camera 2 from world
		cv::Mat R2w = pKF2->GetRotation();
		cv::Mat t2w = pKF2->GetTranslation();

		//Transformation between cameras
		cv::Mat sR12 = s12*R12;
		cv::Mat sR21 = (1.0 / s12)*R12.t();
		cv::Mat t21 = -sR21*t12;

		const std::vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
		const int N1 = vpMapPoints1.size();

		const std::vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
		const int N2 = vpMapPoints2.size();

		std::vector<bool> vbAlreadyMatched1(N1, false);
		std::vector<bool> vbAlreadyMatched2(N2, false);

		for (int i = 0; i<N1; i++)
		{
			MapPoint* pMP = vpMatches12[i];
			if (pMP)
			{
				vbAlreadyMatched1[i] = true;
				int idx2 = pMP->GetIndexInKeyFrame(pKF2);
				if (idx2 >= 0 && idx2<N2)
					vbAlreadyMatched2[idx2] = true;
			}
		}

		std::vector<int> vnMatch1(N1, -1);
		std::vector<int> vnMatch2(N2, -1);

		// Transform from KF1 to KF2 and search
		for (int i1 = 0; i1<N1; i1++)
		{
			MapPoint* pMP = vpMapPoints1[i1];

			if (!pMP || vbAlreadyMatched1[i1])
				continue;

			if (pMP->isBad())
				continue;

			cv::Mat p3Dw = pMP->GetWorldPos();
			cv::Mat p3Dc1 = R1w*p3Dw + t1w;
			cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

			// Depth must be positive
			if (p3Dc2.at<float>(2)<0.0)
				continue;

			const float invz = 1.0 / p3Dc2.at<float>(2);
			const float x = p3Dc2.at<float>(0)*invz;
			const float y = p3Dc2.at<float>(1)*invz;

			const float u = fx*x + cx;
			const float v = fy*y + cy;

			// Point must be inside the image
			if (!pKF2->is_in_image(u, v))
				continue;

			const float maxDistance = pMP->GetMaxDistanceInvariance();
			const float minDistance = pMP->GetMinDistanceInvariance();
			const float dist3D = cv::norm(p3Dc2);

			// Depth must be inside the scale invariance region
			if (dist3D<minDistance || dist3D>maxDistance)
				continue;

			// Compute predicted octave
			const int nPredictedLevel = pMP->PredictScale(dist3D, pKF2);

			// Search in a radius
			const float radius = thRadius*pKF2->mvScaleFactors[nPredictedLevel];

			const std::vector<size_t> vIndices = pKF2->GetFeaturesInArea(u, v, radius);

			if (vIndices.empty())
				continue;

			// Match to the most similar keypoint in the radius
			const cv::Mat dMP = pMP->GetDescriptor();

			int bestDist = INT_MAX;
			int bestIdx = -1;
			for (auto vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
			{
				const size_t idx = *vit;

				const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

				if (kp.octave<nPredictedLevel - 1 || kp.octave>nPredictedLevel)
					continue;

				const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

				const int dist = mpFeatureTracker->DescriptorDistance(dMP, dKF);

				if (dist<bestDist)
				{
					bestDist = dist;
					bestIdx = idx;
				}
			}

			if (bestDist <= mpFeatureTracker->max_descriptor_distance)
			{
				vnMatch1[i1] = bestIdx;
			}
		}

		// Transform from KF2 to KF2 and search
		for (int i2 = 0; i2<N2; i2++)
		{
			MapPoint* pMP = vpMapPoints2[i2];

			if (!pMP || vbAlreadyMatched2[i2])
				continue;

			if (pMP->isBad())
				continue;

			cv::Mat p3Dw = pMP->GetWorldPos();
			cv::Mat p3Dc2 = R2w*p3Dw + t2w;
			cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

			// Depth must be positive
			if (p3Dc1.at<float>(2)<0.0)
				continue;

			const float invz = 1.0 / p3Dc1.at<float>(2);
			const float x = p3Dc1.at<float>(0)*invz;
			const float y = p3Dc1.at<float>(1)*invz;

			const float u = fx*x + cx;
			const float v = fy*y + cy;

			// Point must be inside the image
			if (!pKF1->is_in_image(u, v))
				continue;

			const float maxDistance = pMP->GetMaxDistanceInvariance();
			const float minDistance = pMP->GetMinDistanceInvariance();
			const float dist3D = cv::norm(p3Dc1);

			// Depth must be inside the scale pyramid of the image
			if (dist3D<minDistance || dist3D>maxDistance)
				continue;

			// Compute predicted octave
			const int nPredictedLevel = pMP->PredictScale(dist3D, pKF1);

			// Search in a radius of 2.5*sigma(ScaleLevel)
			const float radius = thRadius*pKF1->mvScaleFactors[nPredictedLevel];

			const std::vector<size_t> vIndices = pKF1->GetFeaturesInArea(u, v, radius);

			if (vIndices.empty())
				continue;

			// Match to the most similar keypoint in the radius
			const cv::Mat dMP = pMP->GetDescriptor();

			int bestDist = INT_MAX;
			int bestIdx = -1;
			for (auto vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
			{
				const size_t idx = *vit;

				const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

				if (kp.octave<nPredictedLevel - 1 || kp.octave>nPredictedLevel)
					continue;

				const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

				const int dist = mpFeatureTracker->DescriptorDistance(dMP, dKF);

				if (dist<bestDist)
				{
					bestDist = dist;
					bestIdx = idx;
				}
			}

			if (bestDist <= mpFeatureTracker->max_descriptor_distance)
			{
				vnMatch2[i2] = bestIdx;
			}
		}

		// Check agreement
		int nFound = 0;

		for (int i1 = 0; i1<N1; i1++)
		{
			int idx2 = vnMatch1[i1];

			if (idx2 >= 0)
			{
				int idx1 = vnMatch2[idx2];
				if (idx1 == i1)
				{
					vpMatches12[i1] = vpMapPoints2[idx2];
					nFound++;
				}
			}
		}

		return nFound;
	}

	int SearchPoints::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12, std::vector<std::pair<size_t, size_t> > &vMatchedPairs, float thRatio, bool bCheckOri) {
		FeatureTracker* mpFeatureTracker = new FlannFeatureTracker(1500);
		const DBoW3::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
		const DBoW3::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

		//Compute epipole in second image
		cv::Mat Cw = pKF1->GetCameraCenter();
		cv::Mat R2w = pKF2->GetRotation();
		cv::Mat t2w = pKF2->GetTranslation();
		cv::Mat C2 = R2w*Cw + t2w;
		const float invz = 1.0f / C2.at<float>(2);
		const float ex = pKF2->fx*C2.at<float>(0)*invz + pKF2->cx;
		const float ey = pKF2->fy*C2.at<float>(1)*invz + pKF2->cy;

		// Find matches between not tracked keypoints
		// Matching speed-up by ORB Vocabulary
		// Compare only ORB that share the same node

		int nmatches = 0;
		std::vector<bool> vbMatched2(pKF2->N, false);
		std::vector<int> vMatches12(pKF1->N, -1);

		std::vector<int> rotHist[HISTO_LENGTH];
		for (int i = 0; i<HISTO_LENGTH; i++)
			rotHist[i].reserve(500);

		const float factor = 1.0f / HISTO_LENGTH;

		DBoW3::FeatureVector::const_iterator f1it = vFeatVec1.begin();
		DBoW3::FeatureVector::const_iterator f2it = vFeatVec2.begin();
		DBoW3::FeatureVector::const_iterator f1end = vFeatVec1.end();
		DBoW3::FeatureVector::const_iterator f2end = vFeatVec2.end();

		while (f1it != f1end && f2it != f2end)
		{
			if (f1it->first == f2it->first)
			{
				for (size_t i1 = 0, iend1 = f1it->second.size(); i1<iend1; i1++)
				{
					const size_t idx1 = f1it->second[i1];

					MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

					// If there is already a MapPoint skip
					if (pMP1)
						continue;

					const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];

					const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

					int bestDist = mpFeatureTracker->min_descriptor_distance;
					int bestIdx2 = -1;
					
					for (size_t i2 = 0, iend2 = f2it->second.size(); i2<iend2; i2++)
					{
						size_t idx2 = f2it->second[i2];

						MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

						// If we have already matched or there is a MapPoint skip
						if (vbMatched2[idx2] || pMP2)
							continue;

						const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

						const int dist = (int)mpFeatureTracker->DescriptorDistance(d1, d2);

						if (dist>mpFeatureTracker->min_descriptor_distance || dist>bestDist)
							continue;

						const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

						{
							const float distex = ex - kp2.pt.x;
							const float distey = ey - kp2.pt.y;
							if (distex*distex + distey*distey<100 * pKF2->mvScaleFactors[kp2.octave])
								continue;
						}

						if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2))
						{
							bestIdx2 = idx2;
							bestDist = dist;
						}
					}

					if (bestIdx2 >= 0)
					{
						const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
						vMatches12[idx1] = bestIdx2;
						nmatches++;

						if (bCheckOri)
						{
							float rot = kp1.angle - kp2.angle;
							if (rot<0.0)
								rot += 360.0f;
							int bin = round(rot*factor);
							if (bin == HISTO_LENGTH)
								bin = 0;
							assert(bin >= 0 && bin<HISTO_LENGTH);
							rotHist[bin].push_back(idx1);
						}
					}
				}

				f1it++;
				f2it++;
			}
			else if (f1it->first < f2it->first)
			{
				f1it = vFeatVec1.lower_bound(f2it->first);
			}
			else
			{
				f2it = vFeatVec2.lower_bound(f1it->first);
			}
		}

		if (bCheckOri)
		{
			int ind1 = -1;
			int ind2 = -1;
			int ind3 = -1;

			ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

			for (int i = 0; i<HISTO_LENGTH; i++)
			{
				if (i == ind1 || i == ind2 || i == ind3)
					continue;
				for (size_t j = 0, jend = rotHist[i].size(); j<jend; j++)
				{
					vMatches12[rotHist[i][j]] = -1;
					nmatches--;
				}
			}

		}

		vMatchedPairs.clear();
		vMatchedPairs.reserve(nmatches);

		for (size_t i = 0, iend = vMatches12.size(); i<iend; i++)
		{
			if (vMatches12[i]<0)
				continue;
			vMatchedPairs.push_back(std::make_pair(i, vMatches12[i]));
		}

		return nmatches;
	}

	int SearchPoints::Fuse(KeyFrame *pKF, const std::vector<MapPoint *> &vpMapPoints, const float th)
	{
		FeatureTracker* mpFeatureTracker = new FlannFeatureTracker(1500);

		cv::Mat Rcw = pKF->GetRotation();
		cv::Mat tcw = pKF->GetTranslation();

		const float &fx = pKF->fx;
		const float &fy = pKF->fy;
		const float &cx = pKF->cx;
		const float &cy = pKF->cy;

		cv::Mat Ow = pKF->GetCameraCenter();

		int nFused = 0;


		for (int i = 0, iend= vpMapPoints.size(); i<iend; i++)
		{
			MapPoint* pMP = vpMapPoints[i];

			if (!pMP || pMP->isBad() || pMP->IsInKeyFrame(pKF))
				continue;

			cv::Mat p3Dw = pMP->GetWorldPos();
			cv::Mat p3Dc = Rcw*p3Dw + tcw;

			// Depth must be positive
			if (p3Dc.at<float>(2)<0.0f)
				continue;

			const float invz = 1 / p3Dc.at<float>(2);
			const float x = p3Dc.at<float>(0)*invz;
			const float y = p3Dc.at<float>(1)*invz;

			const float u = fx*x + cx;
			const float v = fy*y + cy;

			// Point must be inside the image
			if (!pKF->is_in_image(u, v))
				continue;

			//const float ur = u - bf*invz;

			const float maxDistance = pMP->GetMaxDistanceInvariance();
			const float minDistance = pMP->GetMinDistanceInvariance();
			cv::Mat PO = p3Dw - Ow;
			const float dist3D = cv::norm(PO);

			// Depth must be inside the scale pyramid of the image
			if (dist3D<minDistance || dist3D>maxDistance)
				continue;

			// Viewing angle must be less than 60 deg
			cv::Mat Pn = pMP->GetNormal();

			if (PO.dot(Pn)<0.5*dist3D)
				continue;

			int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

			// Search in a radius
			const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

			const std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

			if (vIndices.empty())
				continue;

			// Match to the most similar keypoint in the radius

			const cv::Mat dMP = pMP->GetDescriptor();

			int bestDist = 256;
			int bestIdx = -1;
			for (std::vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
			{
				const size_t idx = *vit;

				const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

				const int &kpLevel = kp.octave;

				if (kpLevel<nPredictedLevel - 1 || kpLevel>nPredictedLevel)
					continue;

				{
					const float &kpx = kp.pt.x;
					const float &kpy = kp.pt.y;
					const float ex = u - kpx;
					const float ey = v - kpy;
					const float e2 = ex*ex + ey*ey;

					if (e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
						continue;
				}

				const cv::Mat &dKF = pKF->mDescriptors.row(idx);

				const int dist = (int)mpFeatureTracker->DescriptorDistance(dMP, dKF);

				if (dist<bestDist)
				{
					bestDist = dist;
					bestIdx = idx;
				}
			}

			// If there is already a MapPoint replace otherwise add new measurement
			if (bestDist <= mpFeatureTracker->min_descriptor_distance)
			{
				MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
				if (pMPinKF)
				{
					if (!pMPinKF->isBad())
					{
						if (pMPinKF->Observations()>pMP->Observations())
							pMP->Replace(pMPinKF);
						else
							pMPinKF->Replace(pMP);
					}
				}
				else
				{
					pMP->AddObservation(pKF, bestIdx);
					pKF->AddMapPoint(pMP, bestIdx);
				}
				nFused++;
			}
		}

		return nFused;
	}

	int SearchPoints::Fuse(KeyFrame *pKF, cv::Mat Scw, const std::vector<MapPoint *> &vpPoints, std::vector<MapPoint *> &vpReplacePoint, float thRadius) {

		FeatureTracker* mpFeatureTracker = new FlannFeatureTracker(1500);

		// Get Calibration Parameters for later projection
		const float &fx = pKF->fx;
		const float &fy = pKF->fy;
		const float &cx = pKF->cx;
		const float &cy = pKF->cy;

		// Decompose Scw
		cv::Mat sRcw = Scw.rowRange(0, 3).colRange(0, 3);
		const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
		cv::Mat Rcw = sRcw / scw;
		cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
		cv::Mat Ow = -Rcw.t()*tcw;

		// Set of MapPoints already found in the KeyFrame
		const std::set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

		int nFused = 0;

		const int nPoints = vpPoints.size();

		// For each candidate MapPoint project and match
		for (int iMP = 0; iMP<nPoints; iMP++)
		{
			MapPoint* pMP = vpPoints[iMP];

			// Discard Bad MapPoints and already found
			if (pMP->isBad() || spAlreadyFound.count(pMP))
				continue;

			// Get 3D Coords.
			cv::Mat p3Dw = pMP->GetWorldPos();

			// Transform into Camera Coords.
			cv::Mat p3Dc = Rcw*p3Dw + tcw;

			// Depth must be positive
			if (p3Dc.at<float>(2)<0.0f)
				continue;

			// Project into Image
			const float invz = 1.0 / p3Dc.at<float>(2);
			const float x = p3Dc.at<float>(0)*invz;
			const float y = p3Dc.at<float>(1)*invz;

			const float u = fx*x + cx;
			const float v = fy*y + cy;

			// Point must be inside the image
			if (!pKF->is_in_image(u, v))
				continue;

			// Depth must be inside the scale pyramid of the image
			const float maxDistance = pMP->GetMaxDistanceInvariance();
			const float minDistance = pMP->GetMinDistanceInvariance();
			cv::Mat PO = p3Dw - Ow;
			const float dist3D = cv::norm(PO);

			if (dist3D<minDistance || dist3D>maxDistance)
				continue;

			// Viewing angle must be less than 60 deg
			cv::Mat Pn = pMP->GetNormal();

			if (PO.dot(Pn)<0.5*dist3D)
				continue;

			// Compute predicted scale level
			const int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

			// Search in a radius
			const float radius = thRadius*pKF->mvScaleFactors[nPredictedLevel];

			const std::vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

			if (vIndices.empty())
				continue;

			// Match to the most similar keypoint in the radius

			const cv::Mat dMP = pMP->GetDescriptor();

			int bestDist = INT_MAX;
			int bestIdx = -1;
			for (auto vit = vIndices.begin(); vit != vIndices.end(); vit++)
			{
				const size_t idx = *vit;
				const int &kpLevel = pKF->mvKeysUn[idx].octave;

				if (kpLevel<nPredictedLevel - 1 || kpLevel>nPredictedLevel)
					continue;

				const cv::Mat &dKF = pKF->mDescriptors.row(idx);

				int dist = mpFeatureTracker->DescriptorDistance(dMP, dKF);

				if (dist<bestDist)
				{
					bestDist = dist;
					bestIdx = idx;
				}
			}

			// If there is already a MapPoint replace otherwise add new measurement
			if (bestDist <= mpFeatureTracker->min_descriptor_distance)
			{
				MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
				if (pMPinKF)
				{
					if (!pMPinKF->isBad())
						vpReplacePoint[iMP] = pMPinKF;
				}
				else
				{
					pMP->AddObservation(pKF, bestIdx);
					pKF->AddMapPoint(pMP, bestIdx);
				}
				nFused++;
			}
		}

		return nFused;
	}

	bool SearchPoints::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame* pKF2)
	{
		// Epipolar line in second image l = x1'F12 = [a b c]
		const float a = kp1.pt.x*F12.at<float>(0, 0) + kp1.pt.y*F12.at<float>(1, 0) + F12.at<float>(2, 0);
		const float b = kp1.pt.x*F12.at<float>(0, 1) + kp1.pt.y*F12.at<float>(1, 1) + F12.at<float>(2, 1);
		const float c = kp1.pt.x*F12.at<float>(0, 2) + kp1.pt.y*F12.at<float>(1, 2) + F12.at<float>(2, 2);

		const float num = a*kp2.pt.x + b*kp2.pt.y + c;

		const float den = a*a + b*b;

		if (den == 0)
			return false;

		const float dsqr = num*num / den;

		return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
	}

	void SearchPoints::ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
	{
		int max1 = 0;
		int max2 = 0;
		int max3 = 0;

		for (int i = 0; i<L; i++)
		{
			const int s = histo[i].size();
			if (s>max1)
			{
				max3 = max2;
				max2 = max1;
				max1 = s;
				ind3 = ind2;
				ind2 = ind1;
				ind1 = i;
			}
			else if (s>max2)
			{
				max3 = max2;
				max2 = s;
				ind3 = ind2;
				ind2 = i;
			}
			else if (s>max3)
			{
				max3 = s;
				ind3 = i;
			}
		}

		if (max2<0.1f*(float)max1)
		{
			ind2 = -1;
			ind3 = -1;
		}
		else if (max3<0.1f*(float)max1)
		{
			ind3 = -1;
		}
	}

	float SearchPoints::RadiusByViewingCos(const float &viewCos)
	{
		if (viewCos>0.998)
			return 2.5;
		else
			return 4.0;
	}
}
