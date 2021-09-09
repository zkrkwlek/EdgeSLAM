#ifndef EDGE_SLAM_VISUALIZER_H
#define EDGE_SLAM_VISUALIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <mutex>

namespace EdgeSLAM {
	class SLAM;
	class Map;
	//class User;
	class Visualizer {
	public:
		Visualizer();
		Visualizer(SLAM* pSystem);
		virtual ~Visualizer();
	public:
		void Init(int w, int h);
		void Run();
		void SetBoolDoingProcess(bool b);
		bool isDoingProcess();
		static void CallBackFunc(int event, int x, int y, int flags, void* userdata);
		void SetMap(Map* pMap);
		Map* GetMap();
		std::string strMapName;
		/*void AddUser(User* pUser);
		void RemoveUser(User* pUser);
		std::vector<User*> GetUsers();*/
		//////////////////////
		////output시각화
	public:
		void ResizeImage(cv::Mat src, cv::Mat& dst);
		void SetOutputImage(cv::Mat out, int type);
		cv::Mat GetOutputImage(int type);
		bool isOutputTypeChanged(int type);
		int mnWindowImgCols, mnWindowImgRows;
		SLAM* mpSystem;
		Map* mpMap;
		int mnVisScale;
		int mnDisplayX, mnDisplayY;
		int mnWidth, mnHeight;
		cv::Mat mVisPoseGraph;
		cv::Point2f mVisMidPt, mVisPrevPt;
		cv::Size mSizeOutputImg;
	private:
		std::vector<cv::Mat> mvOutputImgs;
		std::vector<cv::Rect> mvRects;
		cv::Mat mOutputImage;
		std::vector<bool> mvOutputChanged;
		std::mutex mMutexOutput;
		////output시각화
		//////////////////////

		std::mutex mMutexMap;
		//std::mutex mMutexUserList;
		//std::set<User*> mspUserLists;
		std::mutex mMutexDoingProcess;
		bool mbDoingProcess;
	};
}
#endif