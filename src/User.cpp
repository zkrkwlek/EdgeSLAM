#include <User.h>
#include <Map.h>
#include <Camera.h>
#include <CameraPose.h>
#include <MotionModel.h>

namespace EdgeSLAM {
	User::User():mbMotionModel(false){

	}
	User::User(std::string _user, std::string _map, int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, bool _b) :userName(_user), mapName(_map), mbMapping(_b), mState(UserState::NotEstimated),
		mbProgress(false), mnReferenceKeyFrameID(-1), mnLastKeyFrameID(-1), mnPrevFrameID(-1), mnCurrFrameID(-1), mnLastRelocFrameId(-1), mbMotionModel(false)
	{
		mpMotionModel = new MotionModel();
		mpCamPose = new CameraPose();
		mpCamera = new Camera(_w, _h, _fx, _fy, _cx, _cy, _d1, _d2, _d3, _d4);
		/*mpLastFrame = nullptr;
		SetPose(cv::Mat::eye(3, 3, CV_32FC1), cv::Mat::zeros(3, 1, CV_32FC1));*/
		mpMap = nullptr;
	}
	User::~User() {

	}

	bool mbMotionModel;
	cv::Mat User::GetPosition() {
		return mpCamPose->GetCenter();
	}
	cv::Mat User::GetPose() {
		return mpCamPose->GetPose();
	}
	void User::SetPose(cv::Mat T){
		mpCamPose->SetPose(T);
	}
	cv::Mat User::GetInversePose() {
		return mpCamPose->GetInversePose();
	}
	cv::Mat User::PredictPose(){
		return mpMotionModel->predict();
	}
	void User::UpdatePose(cv::Mat Tnew) {
		mpCamPose->SetPose(Tnew);
		mpMotionModel->update(Tnew);
	}

	UserState User::GetState() {
		std::unique_lock<std::mutex> lock(mMutexState);
		return mState;
	}
	void User::SetState(UserState stat) {
		std::unique_lock<std::mutex> lock(mMutexState);
		mState = stat;
	}
}