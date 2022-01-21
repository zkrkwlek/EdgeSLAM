#ifndef EDGE_SLAM_USER_H
#define EDGE_SLAM_USER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>
#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

namespace EdgeSLAM {

	enum class UserState {
		NoImages, NotEstimated, Success, Failed
	};
	class MotionModel;
	class Frame;
	class KeyFrame;
	class ObjectFrame;
	class Camera;
	class CameraPose;
	class Map;
	
	class User {
	public:
		User();
		User(std::string _user, std::string _map, int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, float _d5, int q, int nskip, bool _b, bool bDeviceTracking = false, bool bimu = false, bool bsave = false, bool bAsync = false);
		virtual ~User();
	public:
		bool mbMotionModel;
		cv::Mat GetPosition();
		cv::Mat GetPose();
		void SetPose(cv::Mat T);
		cv::Mat GetInversePose();
		cv::Mat PredictPose();
		void UpdatePose(cv::Mat Tnew);
		void UpdatePose(cv::Mat Tnew, double ts);
		void UpdateGyro(cv::Mat _R);
		cv::Mat GetGyro();

		cv::Mat GetCameraMatrix();
		cv::Mat GetCameraInverseMatrix();

		ConcurrentVector<cv::Mat> mvDeviceTrajectories;
		ConcurrentVector<double> mvDeviceTimeStamps;
		ConcurrentMap<int, KeyFrame*> KeyFrames; //Frame과 키프레임 연결

	public:
		std::string userName;
		std::string mapName;
		int mnQuality;
		int mnSkip;
		Map* mpMap;
		Camera* mpCamera;
		CameraPose* mpCamPose;
		bool mbMapping, mbIMU, mbDeviceTracking, mbSaveTrajectory, mbAsyncTest;

		Frame* prevFrame;
		std::vector<cv::Mat> vecTrajectories;
		std::vector<double> vecTimestamps;
		ConcurrentMap<int, cv::Mat> mapKeyPoints;
				
		////frame id와 키프레임 id의 대응이 필요함.
		//std::map<int, KeyFrame*> mapKeyFrames;
		KeyFrame* mpRefKF;
		std::atomic<bool> mbProgress;
		std::atomic<int> mnLastKeyFrameID, mnPrevFrameID, mnCurrFrameID, mnLastRelocFrameId;
	
	public:
		UserState GetState();
		void SetState(UserState stat);
	private:
		MotionModel* mpMotionModel;
		UserState mState;
		std::mutex mMutexState;
/////Visual ID
	public:
		void SetVisID(int id);
		int GetVisID();
	private:
		int mnVisID;
		std::mutex mMutexVisID;

		std::mutex mMutexGyro, mMutexAcc;
		cv::Mat Rgyro, tacc;

	};
}

#endif