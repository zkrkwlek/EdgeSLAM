#include <Initializer.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <FeatureTracker.h>
#include <Optimizer.h>
#include <Map.h>
#include <MapPoint.h>
#include <Utils.h>
#include <SearchPoints.h>

namespace EdgeSLAM {
	Initializer::Initializer(int nMinFeatures, int nMinTriangulatedPoints, int nMaxIdDistBetweenIntializingFrames, int nNumOfFailuresAfterWichNumMinTriangulatedPointsIsHalved):
		mnMinFeatures(nMinFeatures), mnMinTriangulatedPoints(nMinTriangulatedPoints), mnMaxIdDistBetweenIntializingFrames(nMaxIdDistBetweenIntializingFrames), mnNumOfFailuresAfterWichNumMinTriangulatedPointsIsHalved(nNumOfFailuresAfterWichNumMinTriangulatedPointsIsHalved), mpRef(nullptr),
		mpInitKeyFrame1(nullptr), mpInitKeyFrame2(nullptr)	{}
	Initializer::~Initializer() {}
	void Initializer::Reset(){
		mpRef = nullptr;
		//stack clear
	}
	void Initializer::Init(Frame* pRef){
		mpRef = pRef;
		mFrameStack.push(mpRef);
	}
	void Initializer::EstimatePose(){
	
	}

	MapState Initializer::InitializeOXR(Frame* pCur, Map* pMap) {
		
		if (mpRef) {
			if (pCur->mnFrameID - mpRef->mnFrameID > mnMaxIdDistBetweenIntializingFrames) {
				ReplaceReferenceFrame();
			}
		}
		mFrameStack.push(pCur);

		cv::Mat MapPoints;
		cv::Mat K = mpRef->K.clone();
		cv::Mat R1 = mpRef->GetRotation();
		cv::Mat t1 = mpRef->GetTranslation();
		cv::Mat R2 = pCur->GetRotation();
		cv::Mat t2 = pCur->GetTranslation();
		cv::Mat F12 = Utils::ComputeF12(R1, t1, R2, t2, K,K);

		mpRef->reset_map_points();
		pCur->reset_map_points();

		auto pRefKeyframe = new KeyFrame(mpRef, pMap);
		auto pCurKeyframe = new KeyFrame(pCur, pMap);

		pRefKeyframe->ComputeBoW();
		pCurKeyframe->ComputeBoW();

		pMap->AddKeyFrame(pRefKeyframe);
		pMap->AddKeyFrame(pCurKeyframe);

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		//	// Search matches that fullfil epipolar constraint
		std::vector<std::pair<size_t, size_t> > vMatchedIndices;

		int nMatch = SearchPoints::SearchForTriangulation(pRefKeyframe, pCurKeyframe, F12, vMatchedIndices);
		std::cout << "OXR = match = " << vMatchedIndices.size() << " " << nMatch << " " << mpRef->mnFrameID << " " << pCur->mnFrameID << std::endl;

		std::vector<cv::Mat> vecMPs(nMatch);
		std::vector<bool> vecInliers(nMatch, false);

		if (nMatch > 100) {

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

			// Triangulate each match
			const int nmatches = vMatchedIndices.size();
			int nMap = 0;
			for (int ikp = 0; ikp < nmatches; ikp++)

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
				
				vecInliers[ikp] = true;
				vecMPs[ikp] = x3D.clone();
				//// Triangulation is succesfull
				MapPoint* pMP = new MapPoint(x3D, pCurKeyframe, pMap, ts);
				pRefKeyframe->AddMapPoint(pMP, idx1);
				pCurKeyframe->AddMapPoint(pMP, idx2);
				pMP->AddObservation(pRefKeyframe, idx1);
				pMP->AddObservation(pCurKeyframe, idx2);
				pMP->ComputeDistinctiveDescriptors();
				pMP->UpdateNormalAndDepth();
				pCur->mvpMapPoints[idx2] = pMP;
				pCur->mvbOutliers[idx2] = false;
				pMap->AddMapPoint(pMP);
				nMap++;
			}//for

			pRefKeyframe->UpdateConnections();
			pCurKeyframe->UpdateConnections();

			if (nMap < 100) {
				//pMap->Delete();
				std::cout << "Map Init fail = " << nMap << std::endl;
				return MapState::NotInitialized;
			}

			
			mpInitKeyFrame1 = pRefKeyframe;
			mpInitKeyFrame2 = pCurKeyframe;
			std::cout << "Map Initialization Success = " << nMap << std::endl;
			return MapState::Initialized;
		}
		else {
			//pMap->Delete();
			return MapState::NotInitialized;
		}
	}

	MapState Initializer::Initialize(Frame* pCur, Map* pMap){
		
		if (mpRef) {
			if (pCur->mnFrameID - mpRef->mnFrameID > mnMaxIdDistBetweenIntializingFrames) {
				ReplaceReferenceFrame();
			}
		}
		mFrameStack.push(pCur);
		//std::cout << "Initialize " << mpRef->mnFrameID << ", " << pCur->mnFrameID << std::endl;

		if (pCur->N < mnMinFeatures || mpRef->N < mnMinFeatures)
			return MapState::NotInitialized;
	
		std::vector<int> idx1, idx2;
		int nMatch = mpFeatureTracker->match(mpRef->mDescriptors, pCur->mDescriptors, idx1, idx2, 0.8);
		std::vector<cv::Point2f> pts1, pts2;
		for (size_t i = 0, iend = idx1.size(); i < iend; i++) {
			int i1 = idx1[i];
			int i2 = idx2[i];
			cv::Point2f pt1 = mpRef->mvKeysUn[i1].pt;
			cv::Point2f pt2 = pCur->mvKeysUn[i2].pt;
			pts1.push_back(pt1);
			pts2.push_back(pt2);
		}
		
		std::vector<uchar> vFInliers;
		cv::Mat Map3D;
		cv::Mat R1, t1;
		cv::Mat E12 = cv::findEssentialMat(pts1, pts2, mpRef->K, cv::FM_RANSAC, 0.999, 1.0, vFInliers);//0.0003
		int nTriangulatedPoints = cv::recoverPose(E12, pts1, pts2, mpRef->K, R1, t1, 50.0, vFInliers, Map3D);
		R1.convertTo(R1, CV_32FC1);
		t1.convertTo(t1, CV_32FC1);

		//ts
		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		long long ts = start.time_since_epoch().count();

		//std::cout << "Map = " << nTriangulatedPoints <<", "<<pts1.size()<<" "<<idx1.size()<<", "<<Map3D.cols<<" "<<vFInliers.size()<< std::endl;
		if (nTriangulatedPoints > mnMinTriangulatedPoints) {
			cv::Mat Tref = cv::Mat::eye(4, 4, CV_32FC1);
			cv::Mat Tcur = cv::Mat::eye(4,4,CV_32FC1);
			R1.copyTo(Tcur.rowRange(0, 3).colRange(0, 3));
			t1.copyTo(Tcur.col(3).rowRange(0, 3));
			
			mpRef->SetPose(Tref);
			pCur->SetPose(Tcur);
			
			mpRef->reset_map_points();
			pCur->reset_map_points();

			auto pRefKeyframe = new KeyFrame(mpRef, pMap);
			auto pCurKeyframe = new KeyFrame(pCur, pMap);

			pRefKeyframe->ComputeBoW();
			pCurKeyframe->ComputeBoW();

			pMap->AddKeyFrame(pRefKeyframe);
			pMap->AddKeyFrame(pCurKeyframe);

			for (int i = 0, iend = vFInliers.size(); i < iend; i++) {
				if (!vFInliers[i])
					continue;
				
				///////////////µª½º°ª Ã¼Å©
				cv::Mat X3D = Map3D.col(i).clone();
				X3D.convertTo(X3D, CV_32FC1);
				X3D /= X3D.at<float>(3);
				if (X3D.at<float>(2) < 0.0) {
					continue;
				}
				///////////////reprojection error
				X3D = X3D.rowRange(0, 3);
				cv::Mat proj1 = X3D.clone();
				cv::Mat proj2 = R1*X3D + t1;
				proj1 = mpRef->K*proj1;
				proj2 = pCur->K*proj2;
				cv::Point2f projected1(proj1.at<float>(0) / proj1.at<float>(2), proj1.at<float>(1) / proj1.at<float>(2));
				cv::Point2f projected2(proj2.at<float>(0) / proj2.at<float>(2), proj2.at<float>(1) / proj2.at<float>(2));
				auto pt1 = pts1[i];
				auto pt2 = pts2[i];
				auto diffPt1 = projected1 - pt1;
				auto diffPt2 = projected2 - pt2;
				float err1 = (diffPt1.dot(diffPt1));
				float err2 = (diffPt2.dot(diffPt2));
				if (err1 > 4.0 || err2 > 4.0)
					continue;
				///////////////reprojection error

				int i1 = idx1[i];
				int i2 = idx2[i];
				
				auto pMP = new MapPoint(X3D, pCurKeyframe, pMap, ts);
				pRefKeyframe->AddMapPoint(pMP, i1);
				pCurKeyframe->AddMapPoint(pMP, i2);

				pMP->AddObservation(pRefKeyframe, i1);
				pMP->AddObservation(pCurKeyframe, i2);

				pMP->ComputeDistinctiveDescriptors();
				pMP->UpdateNormalAndDepth();

				pCur->mvpMapPoints[i2] = pMP;
				pCur->mvbOutliers[i2] = false;

				pMap->AddMapPoint(pMP);
			}

			//std::cout << "KF = " << pRefKeyframe->mnId << ", " << pCurKeyframe->mnId << std::endl;

			pRefKeyframe->UpdateConnections();
			pCurKeyframe->UpdateConnections();

			Optimizer::GlobalBundleAdjustemnt(pMap, 20);

			// Set median depth to 1
			float medianDepth = pRefKeyframe->ComputeSceneMedianDepth(2);
			float invMedianDepth = 1.0f / medianDepth;

			if (medianDepth<0 || pCurKeyframe->TrackedMapPoints(1)<100)
			{
				std::cout << "Wrong initialization, reseting..." << std::endl;
				pMap->Delete();
				return MapState::NotInitialized;
			}
			std::cout << "Map Initialization Success" << std::endl;

			// Scale initial baseline
			cv::Mat Tc2w = pCurKeyframe->GetPose();
			Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3)*invMedianDepth;
			pCurKeyframe->SetPose(Tc2w);
			pCur->SetPose(Tc2w);

			std::vector<MapPoint*> vpAllMapPoints = pCurKeyframe->GetMapPointMatches();
			for (int i = 0; i < vpAllMapPoints.size(); i++) {
				auto pMP = vpAllMapPoints[i];
				if (!pMP)
					continue;
				pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
			}

			mpInitKeyFrame1 = pRefKeyframe;
			mpInitKeyFrame2 = pCurKeyframe;

			/////
			/*
			mpLocalMapper->InsertKeyFrame(pKFini);
			mpLocalMapper->InsertKeyFrame(pKFcur);

			mCurrentFrame.SetPose(pKFcur->GetPose());
			mnLastKeyFrameId=mCurrentFrame.mnId;
			mpLastKeyFrame = pKFcur;

			mvpLocalKeyFrames.push_back(pKFcur);
			mvpLocalKeyFrames.push_back(pKFini);
			mvpLocalMapPoints=mpMap->GetAllMapPoints();
			mpReferenceKF = pKFcur;
			mCurrentFrame.mpReferenceKF = pKFcur;

			mLastFrame = Frame(mCurrentFrame);

			mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

			mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

			mpMap->mvpKeyFrameOrigins.push_back(pKFini);
			*/


			return MapState::Initialized;
		}
		return MapState::NotInitialized;
		
		
	}

	void Initializer::ReplaceReferenceFrame() {
		mpRef = mFrameStack.top();
		mFrameStack.pop();
	}
}