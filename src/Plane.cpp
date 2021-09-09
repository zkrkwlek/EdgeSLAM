#include <Plane.h>
#include <random>
#include <MapPoint.h>

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