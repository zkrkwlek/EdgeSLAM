#include <User.h>
#include <Frame.h>
#include <Map.h>
#include <Camera.h>
#include <CameraPose.h>
#include <MotionModel.h>
#include <MapPoint.h>
#include <KeyFrame.h>

namespace EdgeSLAM {
	User::User():mbMotionModel(false), mpRefKF(nullptr), mnVisID(0), mnUsed(0){

	}
	User::User(std::string _user, std::string _map, int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, float _d5, int q, int nskip, bool _b, bool _bTracking, bool _bBaseLocalMap, bool _bimu, bool _bGBA, bool _bReset, bool _bsave, bool _bAsync) : userName(_user), mapName(_map), mbMapping(_b), mState(UserState::NotEstimated),
		Rgyro(cv::Mat::eye(3, 3, CV_32FC1)), tacc(cv::Mat::zeros(3, 1, CV_32FC1)), mbIMU(_bimu), mbDeviceTracking(_bTracking), mbBaseLocalMap(_bBaseLocalMap), mbSaveTrajectory(_bsave), mnQuality(q), mnSkip(nskip), mbAsyncTest(_bAsync), mbPlaneGBA(_bGBA), mbResetAR(_bReset),
		mbProgress(false), mbRemoved(false), mpRefKF(nullptr), mnUsed(0), mnLastKeyFrameID(-1), mnPrevFrameID(-1), mnCurrFrameID(-1), mnLastRelocFrameId(-1), mbMotionModel(false), mnVisID(0),
		mnDebugTrack(0), mnDebugSeg(0), mnDebugAR(0), mnDebugLabel(0), mnDebugPlane(0)
	{
		mpMotionModel = new MotionModel();
		mpCamPose = new CameraPose();
		mpDevicePose = new CameraPose();
		mpCamera = new Camera(_w, _h, _fx, _fy, _cx, _cy, _d1, _d2, _d3, _d4, _d5);
		/*mpLastFrame = nullptr;
		SetPose(cv::Mat::eye(3, 3, CV_32FC1), cv::Mat::zeros(3, 1, CV_32FC1));*/
		mpMap = nullptr;
	}
	User::~User() {
		//delete mpCamera;
		delete mpCamPose;
		delete mpDevicePose;
		delete mpMotionModel;
		mpMap = nullptr;

		/*auto vecFrames = mapFrames.Get();
		for (auto iter = vecFrames.begin(), iend = vecFrames.end(); iter != iend; iter++) {
			auto frame = iter->second;
			delete frame;
		}
		auto vecObjFrames = objFrames.Get();
		for (auto iter = vecObjFrames.begin(), iend = vecObjFrames.end(); iter != iend; iter++) {
			auto frame = iter->second;
			delete frame;
		}
		mapFrames.Release();
		objFrames.Release();
		*/
		
		auto setKFs = mSetLocalKeyFrames.Get();
		for (auto iter = setKFs.begin(), iend = setKFs.end(); iter != iend; iter++) {
			auto pKF = *iter;
			pKF->mnConnectedDevices--;
		}
		mSetLocalKeyFrames.Release();

		{
			////local mps
			/*auto setMPs = mSetMapPoints.Get();
			for (auto iter = setMPs.begin(), iend = setMPs.end(); iter != iend; iter++) {
				auto pMP = *iter;
				if(pMP->mSetConnected.Count(this))
					pMP->mSetConnected.Erase(this);
			}
			mSetMapPoints.Release();*/
		}
		mvDeviceTimeStamps.Release();
		mvDeviceTrajectories.Release();
		for (int i = 0, iend = vecTrajectories.size(); i < iend; i++)
			vecTrajectories[i].release();
		std::vector<cv::Mat>().swap(vecTrajectories);
		std::vector<double>().swap(vecTimestamps);

		mapKeyPoints.Release();
		KeyFrames.Release();
		ImageDatas.Release();
		QueueNotiMsg.Release();
		//delete mapFrames;
	}

	bool mbMotionModel;
	cv::Mat User::GetPosition() {
		return mpCamPose->GetCenter();
	}

	cv::Mat User::GetDevicePose(){
		return mpDevicePose->GetPose();
	}
	void User::SetDevicePose(cv::Mat T){
		mpDevicePose->SetPose(T);
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
		vecTrajectories.push_back(Tnew);
	}
	void User::UpdatePose(cv::Mat Tnew, double ts) {
		mpCamPose->SetPose(Tnew);
		mpMotionModel->update(Tnew);
		vecTrajectories.push_back(Tnew);
		vecTimestamps.push_back(ts);
	}
	void User::UpdateGyro(cv::Mat _R) {
		std::unique_lock<std::mutex> lock(mMutexGyro);
		Rgyro = _R.clone();
	}
	cv::Mat User::GetGyro(){
		std::unique_lock<std::mutex> lock(mMutexGyro);
		return Rgyro.clone();
	}

	cv::Mat User::GetCameraMatrix() {
		return mpCamera->K;
	}
	cv::Mat User::GetCameraInverseMatrix(){
		return mpCamera->Kinv;
	}
	cv::Mat User::GetDistortionMatrix() {
		return mpCamera->D;
	}

	UserState User::GetState() {
		std::unique_lock<std::mutex> lock(mMutexState);
		return mState;
	}
	void User::SetState(UserState stat) {
		std::unique_lock<std::mutex> lock(mMutexState);
		mState = stat;
	}
	void User::SetVisID(int id){
		std::unique_lock<std::mutex> lock(mMutexVisID);
		mnVisID = id;
	}
	int User::GetVisID(){
		std::unique_lock<std::mutex> lock(mMutexVisID);
		return mnVisID;
	}
}