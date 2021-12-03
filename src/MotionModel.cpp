#include <MotionModel.h>
#include <CameraPose.h>
namespace EdgeSLAM {
	MotionModel::MotionModel():deltaT(cv::Mat::eye(4, 4, CV_32FC1)), covariance(cv::Mat::eye(6, 6, CV_32FC1))
	{
		mpCamPose = new CameraPose();
	}
	MotionModel::MotionModel(cv::Mat _T, cv::Mat cov):covariance(cov), deltaT(cv::Mat::eye(4, 4, CV_32FC1))
	{
		mpCamPose = new CameraPose(_T);
	}
	MotionModel::~MotionModel(){
		delete mpCamPose;
	}
	
	cv::Mat MotionModel::predict(){
		return deltaT*mpCamPose->GetPose();
	}
	void MotionModel::update(cv::Mat Tnew){
		deltaT = Tnew*mpCamPose->GetInversePose();
		mpCamPose->SetPose(Tnew);
	}
	void MotionModel::apply_correction(){
	
	}
}