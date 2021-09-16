#include <Plane.h>
#include <random>
#include <SLAM.h>
#include <User.h>
#include <Map.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <Segmentator.h>

namespace EdgeSLAM {
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

	void PlaneProcessor::EstimateLocalMapPlanes(SLAM* system, Map* map, KeyFrame* pKF) {
		
		if (map->mnNumPlaneEstimation >= 1){
			std::cout << "Already Doing Plane Estimation!! = " << map->mnNumPlaneEstimation << std::endl;
			return;
		}
		else {
			std::cout << "Start PE = " << map->mnNumPlaneEstimation << std::endl;
		}
		map->mnNumPlaneEstimation++;
		LocalMap* pLocalMap = new LocalCovisibilityMap();
		std::vector<MapPoint*> vpLocalMPs;
		std::vector<KeyFrame*> vpLocalKFs;

		//std::cout << "Track::LocalMap::Update::start" << std::endl;
		LocalIndoorModel *pModel = new LocalIndoorModel(pKF);
		pLocalMap->UpdateLocalMap(pKF, vpLocalKFs, vpLocalMPs);

		std::set<MapPoint*> spFloorMPs, spWallMPs, spCeilMPs;
		std::vector<MapPoint*> vpFloorMPs, vpWallMPs, vpCeilMPs;

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
						break;
					case (int)ObjectLabel::WALL:
						if (spWallMPs.count(pMPi))
							continue;
						spWallMPs.insert(pMPi);
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
			map->mnNumPlaneEstimation--;
			return;
		}
		std::cout << "pe = " << spFloorMPs.size() << " " << spWallMPs.size() << std::endl;

		//map->ClearPlanarMPs();
		{
			std::vector<MapPoint*> vpMPs(spFloorMPs.begin(), spFloorMPs.end());
			std::vector<MapPoint*> vpOutlierMPs;
			pModel->mpFloor->mbInit = PlaneProcessor::PlaneInitialization(pModel->mpFloor, vpMPs, vpOutlierMPs);

			/*if (pModel->mpFloor->mbInit) {
				for (int i = 0, iend = pModel->mpFloor->mvpMPs.size(); i < iend; i++) {
					auto pMP = pModel->mpFloor->mvpMPs[i];
					if (!pMP || pMP->isBad())
						continue;
					map->AddPlanarMP(pMP->GetWorldPos(), 0);
				}
			}*/
		}
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