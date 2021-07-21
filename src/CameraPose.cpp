#include <CameraPose.h>

namespace EdgeSLAM {
	CameraPose::CameraPose(){
		SetPose(cv::Mat::eye(4, 4, CV_32FC1));
	}
	CameraPose::CameraPose(cv::Mat T){
		SetPose(T);
	}
	CameraPose::~CameraPose(){}

	void CameraPose::SetPose(cv::Mat T){
		std::unique_lock<std::mutex> lock(mMutexPose);
		Tcw = T.clone();
		Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
		tcw = Tcw.col(3).rowRange(0, 3);
		Ow = -Rcw.t()*tcw;
	}
	cv::Mat CameraPose::GetPose(){
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Tcw.clone();
	}
	void CameraPose::GetPose(cv::Mat& R, cv::Mat& t){
		std::unique_lock<std::mutex> lock(mMutexPose);
		R = Rcw.clone();
		t = tcw.clone();
	}
	cv::Mat CameraPose::GetCenter(){
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Ow.clone();
	}
	cv::Mat CameraPose::GetInversePose(){
		std::unique_lock<std::mutex> lock(mMutexPose);
		cv::Mat Tinv = cv::Mat::eye(4, 4, CV_32FC1);
		cv::Mat rinv = Rcw.t();
		cv::Mat tinv = Ow.clone();
		rinv.copyTo(Tinv.rowRange(0, 3).colRange(0, 3));
		tinv.copyTo(Tinv.col(3).rowRange(0, 3));
		return Tinv.clone();
	}
	cv::Mat CameraPose::GetRotation(){
		std::unique_lock<std::mutex> lock(mMutexPose);
		return Rcw.clone();
	}
	cv::Mat CameraPose::GetTranslation(){
		std::unique_lock<std::mutex> lock(mMutexPose);
		return tcw.clone();
	}
}