#include <Plane.h>
#include <random>
#include <SLAM.h>
#include <User.h>
#include <Map.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <Segmentator.h>
#include <SearchPoints.h>
#include <FeatureTracker.h>
#include <Utils.h>

namespace EdgeSLAM {

	float PlaneProcessor::HoughBinSize = 30.0;
	ConcurrentMap<int, cv::Mat> PlaneProcessor::PlanarHoughImages;
	ConcurrentMap<int, LocalIndoorModel*> PlaneProcessor::LocalPlanarMap;

	float PlaneProcessor::fHistSize = 0.02f;
	int PlaneProcessor::nTrial = 1500;
	float PlaneProcessor::fDistance = 0.02;
	float PlaneProcessor::fRatio = 0.2;
	float PlaneProcessor::fNormal = 0.01;

	Plane::Plane() :mbInit(false), mbParallel(false) {
		matPlaneParam = cv::Mat::zeros(4, 1, CV_32FC1);
		normal = matPlaneParam.rowRange(0, 3);
		distance = matPlaneParam.at<float>(3);
	}
	Plane::~Plane() {

	}
	void Plane::SetParam(cv::Mat m) {
		std::unique_lock<std::mutex> lockTemp(mMutexParam);
		matPlaneParam = cv::Mat::zeros(4, 1, CV_32FC1);
		matPlaneParam = m.clone();

		normal = matPlaneParam.rowRange(0, 3);
		distance = matPlaneParam.at<float>(3);
		norm = normal.dot(normal);
		normal /= norm;
		distance /= norm;
		norm = 1.0;

		normal.copyTo(matPlaneParam.rowRange(0, 3));
		matPlaneParam.at<float>(3) = distance;

	}
	void Plane::SetParam(cv::Mat n, float d) {
		std::unique_lock<std::mutex> lockTemp(mMutexParam);
		matPlaneParam = cv::Mat::zeros(4, 1, CV_32FC1);
		normal = n.clone();
		distance = d;
		norm = normal.dot(normal);
		normal.copyTo(matPlaneParam.rowRange(0, 3));
		matPlaneParam.at<float>(3) = d;
	}
	void Plane::GetParam(cv::Mat& n, float& d) {
		std::unique_lock<std::mutex> lockTemp(mMutexParam);
		n = normal.clone();
		d = distance;
	}
	cv::Mat Plane::GetParam() {
		std::unique_lock<std::mutex> lockTemp(mMutexParam);
		return matPlaneParam.clone();
	}

	LocalIndoorModel::LocalIndoorModel() {
		mpFloor = new Plane();
		mpWall1 = new Plane();
		mpWall2 = new Plane();
		mpWall3 = new Plane();
		mpCeil = new Plane();
	}
	LocalIndoorModel::LocalIndoorModel(KeyFrame* pKF):mpTargetKF(pKF){
		mpFloor = new Plane();
		mpWall1 = new Plane();
		mpWall2 = new Plane();
		mpWall3 = new Plane();
		mpCeil = new Plane();
	}
	LocalIndoorModel::~LocalIndoorModel() {

	}

	cv::Mat PlaneProcessor::CalcPlaneRotationMatrix(cv::Mat normal) {
		//euler zxy
		cv::Mat Nidealfloor = cv::Mat::zeros(3, 1, CV_32FC1);

		Nidealfloor.at<float>(1) = -1.0;
		float nx = normal.at<float>(0);
		float ny = normal.at<float>(1);
		float nz = normal.at<float>(2);

		float d1 = atan2(nx, -ny);
		float d2 = atan2(-nz, sqrt(nx*nx + ny*ny));
		cv::Mat R = Utils::RotationMatrixFromEulerAngles(d1, d2, 0.0, "ZXY");

		return R;
	}
	bool PlaneProcessor::PlaneInitialization2(cv::Mat src, cv::Mat& res, cv::Mat& matInliers, cv::Mat& matOutliers, int ransac_trial, float thresh_distance, float thresh_ratio) {

		//N
		//N-1

		//RANSAC
		int max_num_inlier = 0;
		cv::Mat best_plane_param;
		cv::Mat inlier;

		cv::Mat param, paramStatus;

		//초기 매트릭스 생성
		if (src.rows < 30)
			return false;
		cv::Mat ones = cv::Mat::ones(src.rows, 1, CV_32FC1);
		cv::Mat mMat;
		cv::hconcat(src, ones, mMat);

		std::random_device rn;
		std::mt19937_64 rnd(rn());
		std::uniform_int_distribution<int> range(0, mMat.rows - 1);

		for (int n = 0; n < ransac_trial; n++) {

			cv::Mat arandomPts = cv::Mat::zeros(0, 4, CV_32FC1);
			//select pts
			for (int k = 0; k < 3; k++) {
				int randomNum = range(rnd);
				cv::Mat temp = mMat.row(randomNum).clone();
				arandomPts.push_back(temp);
			}//select

			 //SVD
			cv::Mat X;
			cv::Mat w, u, vt;
			cv::SVD::compute(arandomPts, w, u, vt, cv::SVD::FULL_UV);
			X = vt.row(3).clone();
			cv::transpose(X, X);

			if (!calcUnitNormalVector(X)) {
				//std::cout << "PE::RANSAC_FITTING::UNIT Vector error" << std::endl;
			}
			cv::Mat checkResidual = abs(mMat*X) < thresh_distance;
			//checkResidual = checkResidual / 255;
			int temp_inlier = cv::countNonZero(checkResidual);

			if (max_num_inlier < temp_inlier) {
				max_num_inlier = temp_inlier;
				param = X.clone();
				paramStatus = checkResidual.clone();
			}
		}//trial

		float planeRatio = ((float)max_num_inlier / mMat.rows);

		if (planeRatio > thresh_ratio) {

			for (int i = 0; i < src.rows; i++) {
				int checkIdx = paramStatus.at<uchar>(i);

				cv::Mat temp = src.row(i).clone();
				if (checkIdx == 0) {
					matOutliers.push_back(temp);
				}
				else {
					matInliers.push_back(temp);
				}
			}
			res = param.clone();
			return true;
		}
		else
		{
			//std::cout << "failed" << std::endl;
			return false;
		}
	};
	bool PlaneProcessor::Ransac_fitting(cv::Mat src, cv::Mat& res, cv::Mat& matInliers, cv::Mat& matOutliers, int ransac_trial, float thresh_distance, float thresh_ratio) {

		//RANSAC
		int max_num_inlier = 0;
		cv::Mat best_plane_param;
		cv::Mat inlier;

		cv::Mat param, paramStatus;

		//초기 매트릭스 생성
		if (src.rows < 30)
			return false;
		int nDim = src.cols;
		int nDim2 = src.cols - 1;

		std::random_device rn;
		std::mt19937_64 rnd(rn());
		std::uniform_int_distribution<int> range(0, src.rows - 1);

		for (int n = 0; n < ransac_trial; n++) {

			cv::Mat arandomPts = cv::Mat::zeros(0, nDim, CV_32FC1);
			//select pts
			for (int k = 0; k < nDim2; k++) {
				int randomNum = range(rnd);
				cv::Mat temp = src.row(randomNum).clone();
				arandomPts.push_back(temp);
			}//select

			 //SVD
			cv::Mat X;
			cv::Mat w, u, vt;
			cv::SVD::compute(arandomPts, w, u, vt, cv::SVD::FULL_UV);
			X = vt.row(nDim2).clone();
			cv::transpose(X, X);

			if (!calcUnitNormalVector(X)) {
				std::cout << "PE::RANSAC_FITTING::UNIT Vector error" << std::endl;
			}
			cv::Mat checkResidual = abs(src*X) < thresh_distance;
			//checkResidual = checkResidual / 255;
			int temp_inlier = cv::countNonZero(checkResidual);

			if (max_num_inlier < temp_inlier) {
				max_num_inlier = temp_inlier;
				param = X.clone();
				paramStatus = checkResidual.clone();
			}
		}//trial

		float planeRatio = ((float)max_num_inlier / src.rows);
		//std::cout << "max = " << max_num_inlier <<"::"<<src.rows<< std::endl;
		if (planeRatio > thresh_ratio) {

			for (int i = 0; i < src.rows; i++) {
				int checkIdx = paramStatus.at<uchar>(i);

				cv::Mat temp = src.row(i).clone();
				if (checkIdx == 0) {
					matOutliers.push_back(temp);
				}
				else {
					matInliers.push_back(temp);
				}
			}
			res = param.clone();
			return true;
		}
		else
		{
			//std::cout << "failed" << std::endl;
			return false;
		}
	};

	void PlaneProcessor::CreatePlanarMapPoints(Map* map, KeyFrame* targetKF, cv::Mat param)
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
			cv::Mat paramt = param.t();
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
					cv::Mat A = cv::Mat::zeros(5, 4, CV_32F);
					A.row(0) = xn1.at<float>(0)*Tcw1.row(2) - Tcw1.row(0);
					A.row(1) = xn1.at<float>(1)*Tcw1.row(2) - Tcw1.row(1);
					A.row(2) = xn2.at<float>(0)*Tcw2.row(2) - Tcw2.row(0);
					A.row(3) = xn2.at<float>(1)*Tcw2.row(2) - Tcw2.row(1);
					paramt.copyTo(A.row(4));
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
				//map->AddPlanarMP(x3D, 1);
				//// Triangulation is succesfull
				MapPoint* pMP = new MapPoint(x3D, targetKF, map);

				pMP->AddObservation(targetKF, idx1);
				pMP->AddObservation(pKF2, idx2);

				targetKF->AddMapPoint(pMP, idx1);
				pKF2->AddMapPoint(pMP, idx2);

				pMP->ComputeDistinctiveDescriptors();

				pMP->UpdateNormalAndDepth();

				map->AddMapPoint(pMP);
				map->mlpNewMPs.push_back(pMP);
			}
		}
	}

	void PlaneProcessor::EstimateLocalMapPlanes(SLAM* system, Map* map, KeyFrame* mpTargetKF) {
		
		int nSize = 360 / HoughBinSize;
		cv::Mat hist = cv::Mat::zeros(nSize, nSize, CV_32SC1);
		PlanarHoughImages.Update(mpTargetKF->mnId, hist);

		if (map->mnNumPlaneEstimation >= 1){
			std::cout << "Already Doing Plane Estimation!! = " << map->mnNumPlaneEstimation << std::endl;
			return;
		}
		else {
			std::cout << "Start PE = " << map->mnNumPlaneEstimation << std::endl;
		}
		map->mnNumPlaneEstimation++;

		std::vector<MapPoint*> vpLocalMPs;
		std::vector<KeyFrame*> vpLocalKFs = mpTargetKF->GetBestCovisibilityKeyFrames(10);
		vpLocalKFs.push_back(mpTargetKF);
		std::set<MapPoint*> spMPs;
		for (std::vector<KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			KeyFrame* pKF = *itKF;
			const std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

			for (std::vector<MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
				MapPoint* pMP = *itMP;
				if (!pMP || pMP->isBad() || spMPs.count(pMP))
					continue;
				vpLocalMPs.push_back(pMP);
				spMPs.insert(pMP);
			}
		}
		LocalIndoorModel *pModel = new LocalIndoorModel(mpTargetKF);
		if (vpLocalMPs.size() < 100){
			map->mnNumPlaneEstimation--;
			return;
		}

		std::set<MapPoint*> spFloorMPs, spWallMPs, spCeilMPs;
		std::vector<MapPoint*> vpFloorMPs, vpWallMPs, vpCeilMPs;

		cv::Mat matFloorData = cv::Mat::zeros(0, 3, CV_32FC1);
		cv::Mat matWallData = cv::Mat::zeros(0, 3, CV_32FC1);
		cv::Mat matCeilData = cv::Mat::zeros(0, 3, CV_32FC1);

		for (auto iter = vpLocalMPs.begin(); iter != vpLocalMPs.end(); iter++) {
			auto pMPi = *iter;// ->first;
			if (!pMPi || pMPi->isBad())
				continue;
						
			if (Segmentator::ObjectPoints.Count(pMPi->mnId)) {
				auto obj = Segmentator::ObjectPoints.Get(pMPi->mnId);
				if (obj) {
					int label = obj->GetLabel();
					switch (label) {
					case (int)ObjectLabel::FLOOR:
						if (spFloorMPs.count(pMPi))
							continue;
						spFloorMPs.insert(pMPi);
						matFloorData.push_back(pMPi->GetWorldPos().t());
						break;
					case (int)ObjectLabel::WALL:
						if (spWallMPs.count(pMPi))
							continue;
						spWallMPs.insert(pMPi);
						matWallData.push_back(pMPi->GetWorldPos().t());
						break;
					case (int)ObjectLabel::CEIL:
						break;
					}
				}
			}
			/*cv::Mat x3D = pMPi->GetWorldPos();
			cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
			tpt += mVisMidPt;
			cv::circle(tempVis, tpt, 2, color, -1);*/
		}//for
		//cluster mps from label

		if (spFloorMPs.size() < 200){
			std::cout << "포인트 더 모아야 함 !!!" << spFloorMPs .size()<< std::endl << std::endl << std::endl;
			map->mnNumPlaneEstimation--;
			return;
		}
		std::cout << "pe = " << spFloorMPs.size() << " " << spWallMPs.size() << std::endl;

		//map->ClearPlanarMPs();
		{
			cv::Mat param;
			cv::Mat inliers = cv::Mat::zeros(0, 3, CV_32FC1);
			cv::Mat outliers = cv::Mat::zeros(0, 3, CV_32FC1);
			int nAddNormal = 0;
			pModel->mpFloor->mbInit = PlaneInitialization2(matFloorData, param, inliers, outliers, nTrial, fDistance, fRatio);
						
			if (pModel->mpFloor->mbInit) {

				////임시로
				if (!Segmentator::floorPlane) {
					Segmentator::floorPlane = pModel->mpFloor;
				}


				auto idx = CalcSphericalCoordinate(param.rowRange(0, 3));
				hist.at<int>(idx)++;
				//std::cout<<"PE = "<<param.t()<<std::endl;
				pModel->mpFloor->SetParam(param);
				
				
				{
					////이부분은 올해안에 완성시키기
					//planar 3d test
					//PlaneProcessor::CreatePlanarMapPoints(map, mpTargetKF, param.clone());
				}

			}
		}

		if(pModel->mpFloor->mbInit)
		{
			cv::Mat fparam = pModel->mpFloor->GetParam();
			////wall estimation
			cv::Mat Rsp = CalcPlaneRotationMatrix(fparam).clone();
			cv::Mat tempWallData = matWallData*Rsp;
			cv::Mat wallData2;
			cv::Mat a = tempWallData.col(0);
			cv::Mat b = tempWallData.col(2);
			cv::Mat c = cv::Mat::ones(a.rows, 1, CV_32FC1);
			cv::hconcat(a, b, a);
			cv::hconcat(a, c, wallData2);

			cv::Mat param;
			cv::Mat inliers = cv::Mat::zeros(0, 3, CV_32FC1);
			cv::Mat outliers = cv::Mat::zeros(0, 3, CV_32FC1);
			int nAddNormal = 0;
			bool bWall = Ransac_fitting(wallData2, param, inliers, outliers, nTrial, fDistance, fRatio);
			if (bWall) {
				for (int i = 0, iend = inliers.rows; i <iend; i++) {
					//map->AddPlanarMP(inliers.row(i), 1);
				}
				/*for (int i = 0, iend = pModel->mpFloor->mvpMPs.size(); i < iend; i++) {
				auto pMP = pModel->mpFloor->mvpMPs[i];
				if (!pMP || pMP->isBad())
				continue;
				map->AddPlanarMP(pMP->GetWorldPos(), 0);
				}*/
				cv::Mat param2;
				cv::Mat inliers2 = cv::Mat::zeros(0, 3, CV_32FC1);
				cv::Mat outliers2 = cv::Mat::zeros(0, 3, CV_32FC1);
				bool bWall2 = Ransac_fitting(outliers, param2, inliers2, outliers2, nTrial, fDistance, fRatio);
				if (bWall2) {
					for (int i = 0, iend = inliers2.rows; i <iend; i++) {
						//map->AddPlanarMP(inliers2.row(i), 2);
					}
					/*for (int i = 0, iend = pModel->mpFloor->mvpMPs.size(); i < iend; i++) {
					auto pMP = pModel->mpFloor->mvpMPs[i];
					if (!pMP || pMP->isBad())
					continue;
					map->AddPlanarMP(pMP->GetWorldPos(), 0);
					}*/
				}
				if (bWall) {
					auto idx = CalcSphericalCoordinate(param.rowRange(0, 3));
					hist.at<int>(idx)++;
				}
				if (bWall2) {
					auto idx = CalcSphericalCoordinate(param2.rowRange(0, 3));
					hist.at<int>(idx)++;
				}
			}


			////wall estimation
		}

		////KF 별 히스토그램 갱신 및 이미지로 저장
		//PlanarHoughImages.Update(mpTargetKF->mnId, hist);
		//for (std::vector<KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		//{

		//	KeyFrame* pKF = *itKF;
		//	if (pKF == mpTargetKF)
		//		continue;
		//	if (PlanarHoughImages.Count(pKF->mnId)) {
		//		cv::Mat hist2 = PlanarHoughImages.Get(pKF->mnId);
		//		hist2 += hist;
		//		PlanarHoughImages.Update(pKF->mnId, hist2);
		//		std::stringstream ss;
		//		ss << "../bin/img/hist_" << pKF->mnId << ".jpg";
		//		cv::imwrite(ss.str(), hist2);
		//	}
		//}


		//std::vector<MapPoint*> vpOutlierWallMPs, vpOutlierWallMPs2;
		//{
		//	std::vector<MapPoint*> vpMPs(spWallMPs.begin(), spWallMPs.end());

		//	pModel->mpWall1->mbInit = PlaneProcessor::PlaneInitialization(pModel->mpWall1, vpMPs, vpOutlierWallMPs, 1500, 0.01);
		//	//if (pModel->mpWall1->mbInit) {
		//	//	for (int i = 0, iend = pModel->mpWall1->mvpMPs.size(); i < iend; i++) {
		//	//		auto pMP = pModel->mpWall1->mvpMPs[i];
		//	//		if (!pMP || pMP->isBad())
		//	//			continue;
		//	//		//map->AddPlanarMP(pMP->GetWorldPos(), 1);
		//	//	}
		//	//}
		//}

		map->mnNumPlaneEstimation--;
	}

	cv::Point PlaneProcessor::CalcSphericalCoordinate(cv::Mat normal) {
		cv::Mat a = normal.clone();
		
		cv::Mat b = normal.clone(); //0 y z
		b.at<float>(0) = 0.0;

		cv::Mat c = cv::Mat::zeros(3, 1, CV_32FC1);
		c.at<float>(2) = 1.0; // 0 0 1

		float len_a = sqrt(a.dot(a));
		float len_b = sqrt(b.dot(b));
		float len_c = sqrt(c.dot(c));

		float azi = b.dot(c) / (len_b*len_c) * 180.0 / CV_PI;
		if (azi < 0.0)
			azi += 360.0;
		if (azi >= 360.0)
			azi -= 360.0;
		float ele = b.dot(a) / (len_a*len_b) * 180.0 / CV_PI;
		if (ele < 0.0)
			ele += 360.0;
		if (ele >= 360.0)
			ele -= 360.0;

		int x = (int)azi / HoughBinSize;
		int y = (int)ele / HoughBinSize;

		return cv::Point(x, y);
	}

	bool PlaneProcessor::calcUnitNormalVector(cv::Mat& X) {
		float sum = 0.0;
		int nDim = X.rows - 1;
		for (size_t i = 0; i < nDim; i++) {
			sum += (X.at<float>(i)*X.at<float>(i));
		}
		sum = sqrt(sum);
		if (sum != 0) {
			X /= sum;
			return true;
		}
		return false;
	}
	int PlaneProcessor::GetNormalType(cv::Mat X) {
		float maxVal = 0.0;
		int idx;
		for (int i = 0; i < 3; i++) {
			float val = abs(X.at<float>(i));
			if (val > maxVal) {
				maxVal = val;
				idx = i;
			}
		}
		return idx;
	}
	cv::Mat PlaneProcessor::CreateWorldPoint(cv::Mat Xcam, cv::Mat Tinv, float depth) {
		if (depth <= 0.0) {
		}
		cv::Mat X = Xcam*depth;
		X.push_back(cv::Mat::ones(1, 1, CV_32FC1));
		cv::Mat estimated = Tinv*X;
		return estimated.rowRange(0, 3);
	}
	float PlaneProcessor::CalculateDepth(cv::Mat Xcam, cv::Mat Pinv) {
		float depth = -Pinv.at<float>(3) / Xcam.dot(Pinv.rowRange(0, 3));
		return depth;
	}
	cv::Mat PlaneProcessor::CalcInverPlaneParam(cv::Mat P, cv::Mat Tinv) {
		return Tinv.t()*P;
	}

	bool PlaneProcessor::PlaneInitialization(Plane* plane, std::vector<MapPoint*> vpMPs, std::vector<MapPoint*>& vpOutlierMPs, int ransac_trial, float thresh_distance, float thresh_ratio) {
		//RANSAC
		int max_num_inlier = 0;
		cv::Mat best_plane_param;
		cv::Mat inlier;

		cv::Mat param, paramStatus;

		//초기 매트릭스 생성
		cv::Mat mMat = cv::Mat::zeros(0, 4, CV_32FC1);
		std::vector<int> vIdxs;
		for (int i = 0; i < vpMPs.size(); i++) {
			auto pMP = vpMPs[i];
			if (!pMP || pMP->isBad())
				continue;
			cv::Mat temp = pMP->GetWorldPos();
			temp.push_back(cv::Mat::ones(1, 1, CV_32FC1));
			mMat.push_back(temp.t());
			vIdxs.push_back(i);
		}
		if (mMat.rows < 10)
			return false;
		std::random_device rn;
		std::mt19937_64 rnd(rn());
		std::uniform_int_distribution<int> range(0, mMat.rows - 1);

		for (int n = 0; n < ransac_trial; n++) {

			cv::Mat arandomPts = cv::Mat::zeros(0, 4, CV_32FC1);
			//select pts
			for (int k = 0; k < 3; k++) {
				int randomNum = range(rnd);
				cv::Mat temp = mMat.row(randomNum).clone();
				arandomPts.push_back(temp);
			}//select

			 //SVD
			cv::Mat X;
			cv::Mat w, u, vt;
			cv::SVD::compute(arandomPts, w, u, vt, cv::SVD::FULL_UV);
			X = vt.row(3).clone();
			cv::transpose(X, X);

			calcUnitNormalVector(X);
			//reversePlaneSign(X);

			/*cv::Mat X2 = vt.col(3).clone();
			calcUnitNormalVector(X2);
			reversePlaneSign(X2);
			std::cout << sum(abs(mMatFromMap*X)) << " " << sum(abs(mMatFromMap*X2)) << std::endl;*/

			//cv::Mat checkResidual = abs(mMatCurrMap*X);
			//threshold(checkResidual, checkResidual, thresh_plane_distance, 1.0, cv::THRESH_BINARY_INV);
			cv::Mat checkResidual = abs(mMat*X) < thresh_distance;
			checkResidual = checkResidual / 255;
			int temp_inlier = cv::countNonZero(checkResidual);

			if (max_num_inlier < temp_inlier) {
				max_num_inlier = temp_inlier;
				param = X.clone();
				paramStatus = checkResidual.clone();
			}
		}//trial

		float planeRatio = ((float)max_num_inlier / mMat.rows);

		if (planeRatio > thresh_ratio) {
			cv::Mat tempMat = cv::Mat::zeros(0, 4, CV_32FC1);
			cv::Mat pParam = param.clone();
			/*pPlane->matPlaneParam = pParam.clone();
			pPlane->mnPlaneID = ++nPlaneID;*/

			cv::Mat normal = pParam.rowRange(0, 3);
			float dist = pParam.at<float>(3);
			plane->SetParam(normal, dist);
			//pPlane->norm = sqrt(pPlane->normal.dot(pPlane->normal));

			for (int i = 0; i < mMat.rows; i++) {
				int checkIdx = paramStatus.at<uchar>(i);
				//std::cout << checkIdx << std::endl;
				auto pMP = vpMPs[vIdxs[i]];
				if (checkIdx == 0) {
					vpOutlierMPs.push_back(pMP);
					continue;
				}
				if (pMP && !pMP->isBad()) {
					//평면에 대한 레이블링이 필요함.
					//pMP->SetRecentLayoutFrameID(nTargetID);
					//pMP->SetPlaneID(pPlane->mnPlaneID);
					plane->mvpMPs.push_back(pMP);
					tempMat.push_back(mMat.row(i));
				}
			}
			//평면 정보 생성.

			cv::Mat X;
			cv::Mat w, u, vt;
			cv::SVD::compute(tempMat, w, u, vt, cv::SVD::FULL_UV);
			X = vt.row(3).clone();
			cv::transpose(X, X);
			calcUnitNormalVector(X);
			int idx = GetNormalType(X);
			if (X.at<float>(idx) > 0.0)
				X *= -1.0;
			//pPlane->mnCount = pPlane->mvpMPs.size();
			//std::cout <<"PLANE::"<< planeRatio << std::endl;
			//std::cout << "Update::" << pPlane->matPlaneParam.t() << ", " << X.t() <<", "<<pPlane->mvpMPs.size()<<", "<<nReject<< std::endl;
			plane->SetParam(X.rowRange(0, 3), X.at<float>(3));
			return true;
		}
		else
		{
			//std::cout << "failed" << std::endl;
			return false;
		}
	}
}