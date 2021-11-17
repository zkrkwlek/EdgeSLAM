#ifndef EDGE_SLAM_USER_H
#define EDGE_SLAM_USER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <atomic>
#include <ConcurrentMap.h>

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
		User(std::string _user, std::string _map, int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, float _d5, bool _b, bool bDeviceTracking = false, bool bimu = false);
		virtual ~User();
	public:
		bool mbMotionModel;
		cv::Mat GetPosition();
		cv::Mat GetPose();
		void SetPose(cv::Mat T);
		cv::Mat GetInversePose();
		cv::Mat PredictPose();
		void UpdatePose(cv::Mat Tnew);
		void UpdateGyro(cv::Mat _R);
		cv::Mat GetGyro();

		cv::Mat GetCameraMatrix();
		cv::Mat GetCameraInverseMatrix();
	public:
		std::string userName;
		std::string mapName;
		Map* mpMap;
		Camera* mpCamera;
		CameraPose* mpCamPose;
		bool mbMapping, mbIMU, mbDeviceTracking;

		//std::map<int, Frame*> mapFrames;
		ConcurrentMap<int, Frame*> mapFrames;
		ConcurrentMap<int, ObjectFrame*> objFrames;
		//std::map<int, KeyFrame*> mapKeyFrames;
		std::atomic<bool> mbProgress;
		std::atomic<int> mnReferenceKeyFrameID, mnLastKeyFrameID, mnPrevFrameID, mnCurrFrameID, mnLastRelocFrameId;
	
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
		void AddDevicePosition(cv::Mat pos);
		std::vector<cv::Mat> GetDevicePositions();
	private:
		int mnVisID;
		std::mutex mMutexVisID;

		std::mutex mMutexDevicePositions;
		std::vector<cv::Mat> mVecDevicePositions;

		std::mutex mMutexGyro, mMutexAcc;
		cv::Mat Rgyro, tacc;

	};
}

#endif