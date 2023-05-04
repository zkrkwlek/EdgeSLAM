#include <Visualizer.h>
#include <SLAM.h>
#include <Map.h>
#include <MapPoint.h>
#include <KeyFrame.h>
#include <User.h>
#include <Segmentator.h>

namespace EdgeSLAM {
	Visualizer::Visualizer() {

	}
	Visualizer::Visualizer(SLAM* pSystem):mpSystem(pSystem), mpMap(nullptr), mnVisScale(20), mnDisplayX(0), mnDisplayY(0), mbDoingProcess(false), mnVisMode(0){

	}
	Visualizer::~Visualizer(){}

	int mnMode = 2;
	int mnMaxMode = 3;
	int mnAxis1 = 0;
	int mnAxis2 = 2;

	bool bSaveMap = false;
	bool bLoadMap = false;
	bool bShowOnlyTrajectory = true;

	void SetAxisMode() {
		switch (mnMode) {
		case 0:
			mnAxis1 = 0;
			mnAxis2 = 2;
			break;
		case 1:
			mnAxis1 = 1;
			mnAxis2 = 2;
			break;
		case 2:
			mnAxis1 = 0;
			mnAxis2 = 1;
			break;
		}
	}
	cv::Point2f rectPt;
	void Visualizer::CallBackFunc(int event, int x, int y, int flags, void* userdata)
	{
		float* tempData = (float*)userdata;
		
		if (event == cv::EVENT_LBUTTONDOWN)
		{
			//std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
			tempData[2] = (float)x - tempData[0];
			tempData[3] = (float)y;

			////button interface
			if (tempData[2] < 50.0 && y < 50) {
				bSaveMap = !bSaveMap;
			}
			else if (tempData[2] < 50 && (y >= 50 && y < 100)) {
				//bShowOnlyTrajectory = !bShowOnlyTrajectory;
				bLoadMap = !bLoadMap;
			}
			////button interface
		}
		else if (event == cv::EVENT_LBUTTONUP)
		{
			//std::cout << "Left button of the mouse is released - position (" << x << ", " << y << ")" << std::endl;
			tempData[4] = (float)x - tempData[0];
			tempData[5] = (float)y;
			tempData[6] = tempData[4] - tempData[2];
			tempData[7] = tempData[5] - tempData[3];
		}
		else if (event == cv::EVENT_RBUTTONDOWN) {
			mnMode++;
			mnMode %= mnMaxMode;
			SetAxisMode();
			/*switch (mnMode) {
			case 0:
			mnAxis1 = 0;
			mnAxis2 = 2;
			break;
			case 1:
			mnAxis1 = 1;
			mnAxis2 = 2;
			break;
			case 2:
			mnAxis1 = 0;
			mnAxis2 = 1;
			break;
			}*/

		}
		else if (event == cv::EVENT_MOUSEWHEEL) {

			if (flags & cv::EVENT_FLAG_CTRLKEY) {
				if (flags > 0) {
					//scroll up
					tempData[8] += 0.2;
				}
				else {
					tempData[8] -= 0.2;
				}
			}
			else {
				if (flags > 0) {
					//scroll up
					tempData[1] += 2.0;
				}
				else {
					//scroll down
					tempData[1] -= 2.0;
					if (tempData[1] <= 0.0) {
						tempData[1] = 2.0;
					}
				}
			}

		}
	}

	int nOutputImages = 0;
	void Visualizer::Init(int w, int h){
		mnWidth = w;
		mnHeight = h;

		mVisPoseGraph = cv::Mat(mnHeight * 2, mnWidth * 2, CV_8UC3, cv::Scalar(255, 255, 255));
		rectangle(mVisPoseGraph, cv::Rect(0, 0, 50, 50), cv::Scalar(255, 255, 0), -1);
		rectangle(mVisPoseGraph, cv::Rect(0, 50, 50, 50), cv::Scalar(0, 255, 255), -1);
		mVisMidPt = cv::Point2f(mnHeight, mnWidth);
		mVisPrevPt = mVisMidPt;

		////맵 옆의 4개의 이미지
		//tracking, segmentation, mapping, ??
		cv::Mat leftImg1 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat leftImg2 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat leftImg3 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat leftImg4 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);

		//right image
		cv::Mat rightImg1 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat rightImg2 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat rightImg3 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);
		cv::Mat rightImg4 = cv::Mat::zeros(mnHeight / 2, mnWidth / 2, CV_8UC3);

		mSizeOutputImg = leftImg1.size();
		//맵
		cv::Mat mapImage = cv::Mat::zeros(mnHeight * 2, mnWidth * 2, CV_8UC3);

		//////sliding window
		//mnWindowImgRows = 4;
		//int nWindowSize = 8;//mpMap->mnMaxConnectedKFs + mpMap->mnHalfConnectedKFs + mpMap->mnQuarterConnectedKFs;
		//mnWindowImgCols = nWindowSize / mnWindowImgRows;
		//if (nWindowSize % 4 != 0)
		//	mnWindowImgCols++;
		//cv::Mat kfWindowImg = cv::Mat::zeros(mnWindowImgRows*mnHeight / 2, mnWindowImgCols * mnWidth / 2, CV_8UC3);

		//0 1 2 3
		mvOutputImgs.push_back((leftImg1));
		cv::Rect r1(0, 0, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r1);
		mvOutputImgs.push_back((leftImg2));
		cv::Rect r2(0, leftImg1.rows, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r2);
		mvOutputImgs.push_back((leftImg3));
		cv::Rect r3(0, leftImg1.rows * 2, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r3);
		mvOutputImgs.push_back((leftImg4));
		cv::Rect r4(0, leftImg1.rows * 3, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r4);

		//right image
		int colRight = leftImg1.cols + mapImage.cols;
		mvOutputImgs.push_back((rightImg1));
		cv::Rect r5(colRight, 0, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r5);
		mvOutputImgs.push_back((rightImg2));
		cv::Rect r6(colRight, leftImg1.rows, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r6);
		mvOutputImgs.push_back((rightImg3));
		cv::Rect r7(colRight, leftImg1.rows * 2, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r7);
		mvOutputImgs.push_back((rightImg4));
		cv::Rect r8(colRight, leftImg1.rows * 3, leftImg1.cols, leftImg1.rows);
		mvRects.push_back(r8);

		//4
		nOutputImages = mvOutputImgs.size();
		mvOutputImgs.push_back((mapImage));
		cv::Rect rMap(leftImg1.cols, 0, mapImage.cols, mapImage.rows);
		mvRects.push_back(rMap);

		////5 윈도우이미지
		//mvOutputImgs.push_back((kfWindowImg));
		//cv::Rect rWindow(leftImg1.cols + mapImage.cols, 0, kfWindowImg.cols, kfWindowImg.rows);
		//mvRects.push_back(rWindow);

		mvOutputChanged = std::vector<bool>(mvRects.size(), false);
		//map

		rectPt = cv::Point2f(r3.x, r3.y);
		int nDisRows = mnHeight * 2;
		int nDisCols = leftImg1.cols + mapImage.cols + rightImg1.cols;// +kfWindowImg.cols;
		mOutputImage = cv::Mat::zeros(nDisRows, nDisCols, CV_8UC3);

		
	}

	
	void Visualizer::Run(){

		//floor plan
		cv::Mat wean = cv::imread("../bin/data/weanhall2.png", cv::IMREAD_COLOR);
		//cv::resize(wean, wean, cv::Size(wean.cols * 1.2, wean.rows * 1.2));
		{
			/*cv::Mat img_gray;
			cv::cvtColor(wean, img_gray, cv::COLOR_BGR2GRAY);
			cv::imshow("ca", img_gray); cv::waitKey(1);
			std::cout <<wean.type()<<" "<<CV_8UC3<<" "<<wean.channels()<<" "<< img_gray.type()<<" "<<CV_8UC1<< "?????" << std::endl;
			cv::Mat img_canny;
			cv::Canny(img_gray, img_canny, 150, 255);

			std::vector<cv::Vec4i> linesP;
			cv::HoughLinesP(img_canny, linesP, 1, (CV_PI / 180), 50, 50, 10);

			cv::Mat img_lane;
			threshold(img_canny, img_lane, 150, 255, cv::THRESH_MASK);

			for (size_t i = 0; i < linesP.size(); i++)
			{
				cv::Vec4i l = linesP[i];
				cv::line(img_lane, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar::all(255), 1, 8);
			}
			cv::imshow("img_lane", img_lane); cv::waitKey(1);*/
		}

		SetAxisMode();
		int nMapImageID = 0;
		
		std::stringstream ss;
		ss << "Output::Display::" << strMapName;
		std::string strWindowName = ss.str();

		//현재 여기서 시각화가 안됨
		//원인 찾는 중
		cv::imshow(strWindowName, mOutputImage);
		cv::moveWindow(strWindowName, mnDisplayX, mnDisplayY);
		float mapControlData[9] = { 0.0, };
		mapControlData[8] = -90.0;
		mapControlData[0] = mnWidth;
		mapControlData[1] = mnVisScale;
		cv::setMouseCallback(strWindowName, EdgeSLAM::Visualizer::CallBackFunc, (void*)mapControlData);

		std::vector<cv::Scalar> planeColors(3);
		planeColors[0] = cv::Scalar(125, 0, 0);
		planeColors[1] = cv::Scalar(0, 125, 0);
		planeColors[2] = cv::Scalar(0, 0, 125);

		std::vector<cv::Scalar> userColors(6);
		userColors[0] = cv::Scalar(255, 255, 0);
		userColors[1] = cv::Scalar(0, 255, 255);
		userColors[2] = cv::Scalar(255, 0, 255);
		userColors[3] = cv::Scalar(255, 0, 0);
		userColors[4] = cv::Scalar(0, 255, 0);
		userColors[5] = cv::Scalar(0, 0, 255);
		

		while (true) {
			if (bSaveMap) {
				//auto vpUsers = GetUsers();
				auto user = mpSystem->GetAllUsersInMap(strMapName)[0];
				mpSystem->pool->EnqueueJob(Segmentator::ProcessPlanarModeling, mpSystem, user);
				bSaveMap = false;
			}
			
			//////Update Visualizer 
			mVisMidPt += cv::Point2f(mapControlData[6], mapControlData[7]);
			mapControlData[6] = 0;
			mapControlData[7] = 0;
			mnVisScale = mapControlData[1];
			float radian = mapControlData[8]* CV_PI/180.0;
			cv::Mat T = cv::Mat::eye(2, 2, CV_32FC1);
			float c = std::cosf(radian);
			float s = std::sinf(radian);
			T.at<float>(0, 0) = c;
			T.at<float>(0, 1) = -s;
			T.at<float>(1, 0) = s;
			T.at<float>(1, 1) = c;

			cv::Mat tempVis = mVisPoseGraph.clone();
			//wean.copyTo(tempVis(cv::Rect(300, 500, wean.cols, wean.rows))); //floor plan

			auto pMap = GetMap();
			if (pMap) {
				{
					////맵포인트 시각화
					auto mmpMap = pMap->GetAllMapPoints();
					for (auto iter = mmpMap.begin(); iter != mmpMap.end(); iter++) {
						auto pMPi = *iter;// ->first;
						if (!pMPi || pMPi->isBad())
							continue;

						//cv::Scalar color = cv::Scalar(0, 0, 0);
						/*if (Segmentator::ObjectPoints.Count(pMPi->mnId)) {
							auto obj = Segmentator::ObjectPoints.Get(pMPi->mnId);
							if (obj) {
								int label = obj->GetLabel()+1;
								color = Segmentator::mvObjectLabelColors[label];
							}
						}*/
						int label = pMPi->mnLabelID;
						cv::Scalar color = Segmentator::mvObjectLabelColors[label];

						cv::Mat x3D = pMPi->GetWorldPos();
						cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
						cv::Mat tempPt(tpt);
						cv::Mat aaa = T*tempPt;
						tpt.x = aaa.at<float>(0);
						tpt.y = aaa.at<float>(1);
						tpt += mVisMidPt;
						cv::circle(tempVis, tpt, 1, color, -1);

						if (pMPi->mSetConnected.Size() > 0) {
							cv::circle(tempVis, tpt, 3, cv::Scalar(255,255,0), 1);
						}

					}
					std::vector<MapPoint*>().swap(mmpMap);
				}
				{
					auto KFs = pMap->GetAllKeyFrames();
					for (int i = 0, iend = KFs.size(); i < iend; i++) {
						auto pKFi = KFs[i];
						cv::Mat Ow = pKFi->GetCameraCenter();
						cv::Scalar color = cv::Scalar(255, 0, 0);
						if (pKFi->mnConnectedDevices > 1) {  
							color.val[2] = 255;
						}
						else if (pKFi->mnConnectedDevices > 0) {
							color.val[1] = 255;
						}
						cv::Point2f tpt = cv::Point2f(Ow.at<float>(mnAxis1) * mnVisScale, Ow.at<float>(mnAxis2) * mnVisScale);
						cv::Mat tempPt(tpt);
						cv::Mat aaa = T*tempPt;
						tpt.x = aaa.at<float>(0);
						tpt.y = aaa.at<float>(1);
						tpt += mVisMidPt;
						cv::circle(tempVis, tpt, 4, color, 1);
					}
				}
				{
					{
						std::map<int, cv::Mat> contentDatas;
						if (mpSystem->TemporalDatas2.Count("OXRMAP"))
							contentDatas = mpSystem->TemporalDatas2.Get("OXRMAP");
						for (auto jter = contentDatas.begin(), jend = contentDatas.end(); jter != jend; jter++) {
							int id = jter->first;
							auto x3D = contentDatas[id];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T * tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;
							cv::circle(tempVis, tpt, 4, cv::Scalar(0, 255, 255), -1);
						}
					}
					{
						std::map<int, cv::Mat> contentDatas;
						if (mpSystem->TemporalDatas2.Count("marker"))
							contentDatas = mpSystem->TemporalDatas2.Get("marker");
						for (auto jter = contentDatas.begin(), jend = contentDatas.end(); jter != jend; jter++) {
							int id = jter->first;
							auto x3D = contentDatas[id];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T*tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;
							cv::circle(tempVis, tpt, 4, cv::Scalar(0, 255, 255), -1);
						}
					}
					{
						std::map<int, cv::Mat> contentDatas;
						if (mpSystem->TemporalDatas2.Count("GBAFloor"))
							contentDatas = mpSystem->TemporalDatas2.Get("GBAFloor");
						for (auto jter = contentDatas.begin(), jend = contentDatas.end(); jter != jend; jter++) {
							int id = jter->first;
							auto x3D = contentDatas[id];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T*tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;
							cv::circle(tempVis, tpt, 3, cv::Scalar(0, 0, 255),-1);
						}
					}
					{
						std::map<int, cv::Mat> contentDatas;
						if (mpSystem->TemporalDatas2.Count("GBAWall"))
							contentDatas = mpSystem->TemporalDatas2.Get("GBAWall");
						for (auto jter = contentDatas.begin(), jend = contentDatas.end(); jter != jend; jter++) {
							int id = jter->first;
							auto x3D = contentDatas[id];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T*tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;
							cv::circle(tempVis, tpt, 3, cv::Scalar(255, 0, 0), -1);
						}
					}
					{
						std::map<int, cv::Mat> contentDatas;
						if (mpSystem->TemporalDatas2.Count("GBACeil"))
							contentDatas = mpSystem->TemporalDatas2.Get("GBACeil");
						for (auto jter = contentDatas.begin(), jend = contentDatas.end(); jter != jend; jter++) {
							int id = jter->first;
							auto x3D = contentDatas[id];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T*tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;
							cv::circle(tempVis, tpt, 3, cv::Scalar(0, 255, 0), -1);
						}
					}
					{
						std::map<int, cv::Mat> contentDatas;
						if (mpSystem->TemporalDatas2.Count("GBAFloorOutlier"))
							contentDatas = mpSystem->TemporalDatas2.Get("GBAFloorOutlier");
						for (auto jter = contentDatas.begin(), jend = contentDatas.end(); jter != jend; jter++) {
							int id = jter->first;
							auto x3D = contentDatas[id];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T*tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;
							cv::circle(tempVis, tpt, 4, cv::Scalar(125, 125, 125),-1);
						}
					}
					
					
					{
						std::map<int, cv::Mat> contentDatas;
						if (mpSystem->TemporalDatas2.Count("path"))
							contentDatas = mpSystem->TemporalDatas2.Get("path");
						for (auto jter = contentDatas.begin(), jend = contentDatas.end(); jter != jend; jter++) {
							int id = jter->first;
							auto temp = contentDatas[id];
							auto x1 = temp.rowRange(0,3);
							auto x2 = temp.rowRange(3,6);
							
							cv::Point2f tpt1 = cv::Point2f(x1.at<float>(mnAxis1) * mnVisScale, x1.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt1(tpt1);
							cv::Mat aaa = T*tempPt1;
							tpt1.x = aaa.at<float>(0);
							tpt1.y = aaa.at<float>(1);
							tpt1 += mVisMidPt;
							
							cv::Point2f tpt2 = cv::Point2f(x2.at<float>(mnAxis1) * mnVisScale, x2.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt2(tpt2);
							cv::Mat bbb = T*tempPt2;
							tpt2.x = bbb.at<float>(0);
							tpt2.y = bbb.at<float>(1);
							tpt2 += mVisMidPt;
							cv::line(tempVis, tpt1, tpt2, cv::Scalar(255, 255, 0), 2);
							//cv::circle(tempVis, tpt, 4, cv::Scalar(0, 255, 255), -1);
						}
					}

					{
						std::map<int, cv::Mat> contentDatas;
						if (mpSystem->TemporalDatas2.Count("pathpos"))
							contentDatas = mpSystem->TemporalDatas2.Get("pathpos");
						for (auto jter = contentDatas.begin(), jend = contentDatas.end(); jter != jend; jter++) {
							int id = jter->first;
							auto x3D = contentDatas[id];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T*tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;
							cv::circle(tempVis, tpt, 6, cv::Scalar(0, 255, 0));
						}
						//std::cout << "Path test = " << contentDatas.size() << std::endl;
					}

					{
						std::map<int, cv::Mat> contentDatas;
						if (mpSystem->TemporalDatas2.Count("content"))
							contentDatas = mpSystem->TemporalDatas2.Get("content");
						for (auto jter = contentDatas.begin(), jend = contentDatas.end(); jter != jend; jter++) {
							int id = jter->first;
							auto x3D = contentDatas[id];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T*tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;
							cv::circle(tempVis, tpt, 4, cv::Scalar(0, 255, 255), -1);
						}
					}

					{
						std::map<int, cv::Mat> contentDatas;
						if (mpSystem->TemporalDatas2.Count("MovingObject"))
							contentDatas = mpSystem->TemporalDatas2.Get("MovingObject");
						for (auto jter = contentDatas.begin(), jend = contentDatas.end(); jter != jend; jter++) {
							int id = jter->first;
							auto x3D = contentDatas[id];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T*tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;
							cv::circle(tempVis, tpt, 6, cv::Scalar(0, 0, 0), -1);
						}
						//std::cout << "Path test = " << contentDatas.size() << std::endl;
					}

					/*std::map<int, cv::Mat> ARFoundationMPs;
					if (mpSystem->TemporalDatas2.Count("ARFoundationMPs"))
						ARFoundationMPs = mpSystem->TemporalDatas2.Get("ARFoundationMPs");
					for (auto jter = ARFoundationMPs.begin(), jend = ARFoundationMPs.end(); jter != jend; jter++) {
						int id = jter->first;
						auto x3Ds = ARFoundationMPs[id];
						for (int j = 0; j < x3Ds.cols; j++) {
							cv::Mat x3D = x3Ds.col(j);
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T*tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;
							cv::circle(tempVis, tpt, 4, cv::Scalar(0, 255, 255), -1);
						}
					}*/
					/*std::map<int, cv::Mat> labelDatas;
					if (mpSystem->TemporalDatas2.Count("label"))
						labelDatas = mpSystem->TemporalDatas2.Get("label");
					std::map<int, cv::Mat> mapDatas;
					if (mpSystem->TemporalDatas2.Count("map"))
						mapDatas = mpSystem->TemporalDatas2.Get("map");
					for (auto jter = mapDatas.begin(), jend = mapDatas.end(); jter != jend; jter++) {
						int id = jter->first;
						if (labelDatas.count(id) == 0 || mapDatas.count(id) == 0)
							continue;
						auto x3D = mapDatas[id];
						auto label = labelDatas[id].at<uchar>(0);
						
						cv::Scalar color = Segmentator::mvObjectLabelColors[label];
						cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
						cv::Mat tempPt(tpt);
						cv::Mat aaa = T*tempPt;
						tpt.x = aaa.at<float>(0);
						tpt.y = aaa.at<float>(1);
						tpt += mVisMidPt;
						cv::circle(tempVis, tpt, 1, color, -1);
					}*/
				}
				{
					////벽데이터 시각화

					if (mpSystem->TemporalDatas.Count("floor")) {
						auto vecDatas = mpSystem->TemporalDatas.Get("floor");
						for (int j = 0; j < vecDatas.size(); j++) {
							cv::Mat x3D = vecDatas[j];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T*tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;

							cv::circle(tempVis, tpt, 3, planeColors[0], -1);
						}
					}

					if (mpSystem->TemporalDatas.Count("wall")) {
						auto vecDatas = mpSystem->TemporalDatas.Get("wall"); 
						for (int j = 0; j < vecDatas.size(); j++) {
							cv::Mat x3D = vecDatas[j];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							cv::Mat tempPt(tpt);
							cv::Mat aaa = T*tempPt;
							tpt.x = aaa.at<float>(0);
							tpt.y = aaa.at<float>(1);
							tpt += mVisMidPt;

							cv::circle(tempVis, tpt, 3, planeColors[1], -1);
						}
					}

					////벽데이터 시각화
					//for (int i = 0, iend = 3; i < iend; i++) {
					//	//auto vMPs = pMap->GetPlanarMPs(i);
					//	auto vMPs = pMap->GetDepthMPs();
					//	for (int j = 0; j < vMPs.size(); j++) {
					//		cv::Mat x3D = vMPs[j];
					//		cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
					//		tpt += mVisMidPt;
					//		cv::circle(tempVis, tpt, 4, planeColors[i], -1);
					//	}
					//}

					/*cv::Mat objMap = pMap->mObjectTest.Get().clone();
					for (int i = 0; i < objMap.rows; i++) {
						cv::Mat x3D = objMap.row(i).colRange(0,3);
						int label = (int)objMap.at<float>(i, 3);
						cv::Scalar color = Segmentator::mvObjectLabelColors[label];
						cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
						tpt += mVisMidPt;
						cv::circle(tempVis, tpt, 4, color, -1);
					}
					objMap.release();*/
				}
			}

			{
				////User 위치 시각화
				//추후 유저별 정보로 변경
				auto vpUsers = mpSystem->GetAllUsersInMap(strMapName);
				for (size_t i = 0, iend = vpUsers.size(); i < iend; i++) {
					auto user = vpUsers[i];
					if (!user)
						continue;
					user->mnUsed++;
					//if (i == 0) {
					//	////gyro testa
					//	auto Ra = user->GetGyro();
					//	cv::Mat Tgyro = cv::Mat::eye(4, 4, CV_32FC1);
					//	Ra.copyTo(Tgyro.rowRange(0, 3).colRange(0, 3));
					//	cv::Mat T = user->GetPose();
					//	T = Tgyro*T;
					//	cv::Mat Rtemp = T.rowRange(0, 3).colRange(0, 3);


					//	cv::Point2f pt1 = cv::Point2f(0,0);
					//	pt1 += mVisMidPt;
					//	cv::circle(tempVis, pt1, 3, cv::Scalar(0, 0, 255), -1);

					//	cv::Mat directionZ = Rtemp.row(2);
					//	cv::Point2f dirPtZ = cv::Point2f(directionZ.at<float>(mnAxis1)* mnVisScale / 10.0, directionZ.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
					//	cv::line(tempVis, pt1, dirPtZ, cv::Scalar(255, 0, 0), 2);

					//	cv::Mat directionY = Rtemp.row(1);
					//	cv::Point2f dirPtY = cv::Point2f(directionY.at<float>(mnAxis1)* mnVisScale / 10.0, directionY.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
					//	cv::line(tempVis, pt1, dirPtY, cv::Scalar(0, 255, 0), 2);

					//	cv::Mat directionX = Rtemp.row(0);
					//	cv::Point2f dirPtX1 = pt1 + cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
					//	cv::Point2f dirPtX2 = pt1 - cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
					//	cv::line(tempVis, dirPtX1, dirPtX2, cv::Scalar(0, 0, 255), 2);

					//}

					{
						////device testa
						cv::Mat T2 = user->GetPose();
						cv::Mat R = T2.rowRange(0, 3).colRange(0, 3);
						cv::Mat t = T2.col(3).rowRange(0, 3);
						auto pos = user->GetPosition();
						cv::Point2f pt1 = cv::Point2f(pos.at<float>(mnAxis1)* mnVisScale, pos.at<float>(mnAxis2)* mnVisScale);
						cv::Mat tempPt(pt1);
						cv::Mat aaa = T*tempPt;
						pt1.x = aaa.at<float>(0);
						pt1.y = aaa.at<float>(1);
						pt1 += mVisMidPt;
						cv::circle(tempVis, pt1, 3, cv::Scalar(0, 0, 255), -1);

						cv::Mat directionZ = R.row(2);
						cv::Point2f dirPtZ = cv::Point2f(directionZ.at<float>(mnAxis1)* mnVisScale / 10.0, directionZ.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
						cv::line(tempVis, pt1, dirPtZ, cv::Scalar(255, 0, 0), 2);

						cv::Mat directionY = R.row(1);
						cv::Point2f dirPtY = cv::Point2f(directionY.at<float>(mnAxis1)* mnVisScale / 10.0, directionY.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
						cv::line(tempVis, pt1, dirPtY, cv::Scalar(0, 255, 0), 2);

						cv::Mat directionX = R.row(0);
						cv::Point2f dirPtX1 = pt1 + cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
						//cv::Point2f dirPtX2 = pt1 - cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
						cv::line(tempVis, pt1, dirPtX1, cv::Scalar(0, 0, 255), 2);
						std::stringstream ss;
						ss << user->userName;
						cv::putText(tempVis, ss.str(), cv::Point(pt1.x, pt1.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255),2);
					}

					//{
					//	////arfoudation test
					//	cv::Mat T2 = user->GetDevicePose();
					//	cv::Mat R = T2.rowRange(0, 3).colRange(0, 3);
					//	cv::Mat t = T2.col(3).rowRange(0, 3);
					//	cv::Mat pos = -R.t()*t;
					//	//auto pos = user->GetPosition();
					//	cv::Point2f pt1 = cv::Point2f(pos.at<float>(mnAxis1)* mnVisScale, pos.at<float>(mnAxis2)* mnVisScale);
					//	cv::Mat tempPt(pt1);
					//	cv::Mat aaa = T*tempPt;
					//	pt1.x = aaa.at<float>(0);
					//	pt1.y = aaa.at<float>(1);
					//	pt1 += mVisMidPt;
					//	cv::circle(tempVis, pt1, 5, cv::Scalar(0, 0, 0), -1);

					//	cv::Mat directionZ = R.row(2);
					//	cv::Point2f dirPtZ = cv::Point2f(directionZ.at<float>(mnAxis1)* mnVisScale / 10.0, directionZ.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
					//	cv::line(tempVis, pt1, dirPtZ, cv::Scalar(255, 0, 0), 2);

					//	cv::Mat directionY = R.row(1);
					//	cv::Point2f dirPtY = cv::Point2f(directionY.at<float>(mnAxis1)* mnVisScale / 10.0, directionY.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
					//	cv::line(tempVis, pt1, dirPtY, cv::Scalar(0, 255, 0), 2);

					//	cv::Mat directionX = R.row(0);
					//	cv::Point2f dirPtX1 = pt1 + cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
					//	//cv::Point2f dirPtX2 = pt1 - cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
					//	cv::line(tempVis, dirPtX1, pt1, cv::Scalar(255, 0, 255), 2);
					//}
					
					auto pos = user->GetPosition();
					cv::Point2f pt1 = cv::Point2f(pos.at<float>(mnAxis1)* mnVisScale, pos.at<float>(mnAxis2)* mnVisScale);
					cv::Mat tempPt(pt1);
					cv::Mat aaa = T*tempPt;
					pt1.x = aaa.at<float>(0);
					pt1.y = aaa.at<float>(1);
					
					pt1 += mVisMidPt;
					if(user->mbMapping)
						cv::circle(tempVis, pt1, 4, cv::Scalar(0, 0, 255), -1);
					else {
						cv::circle(tempVis, pt1, 4, userColors[i], -1);
					}
					//auto vecTrajectories = user->mvDeviceTrajectories.get();
					//for (int j = 0; j < vecTrajectories.size(); j += 1) {

					//	cv::Mat R = vecTrajectories[j].rowRange(0, 3);
					//	cv::Mat t = vecTrajectories[j].row(3).t();
					//	t = -R.t()*t;  //camera center
					//	cv::Point2f pt1 = cv::Point2f(t.at<float>(mnAxis1)* mnVisScale, t.at<float>(mnAxis2)* mnVisScale);
					//	pt1 += mVisMidPt;
					//	cv::circle(tempVis, pt1, 1, userColors[i], -1);
					//}
					user->mnUsed--;
				}
				std::vector<User*>().swap(vpUsers);
			}

			SetOutputImage(tempVis, nOutputImages);
			////////Update Map Visualizer
			int N = mvOutputImgs.size();
			for (int i = 0; i < N; i++) {
				if (isOutputTypeChanged(i)) {
					cv::Mat mTrackImg = GetOutputImage(i);
					mTrackImg.copyTo(mOutputImage(mvRects[i]));
				}
			}
			/*if (isOutputTypeChanged(0)) {
				cv::Mat mTrackImg = GetOutputImage(0);
				mTrackImg.copyTo(mOutputImage(mvRects[0]));
			}
			if (isOutputTypeChanged(1)) {
				cv::Mat mWinImg = GetOutputImage(1);
				mWinImg.copyTo(mOutputImage(mvRects[1]));
			}
			if (isOutputTypeChanged(2)) {
				cv::Mat mMapImg = GetOutputImage(2);
				mMapImg.copyTo(mOutputImage(mvRects[2]));
			}
			if (isOutputTypeChanged(3)) {
				cv::Mat mMappingImg = GetOutputImage(3);
				mMappingImg.copyTo(mOutputImage(mvRects[3]));
			}
			if (isOutputTypeChanged(4)) {
				cv::Mat mMappingImg = GetOutputImage(4);
				mMappingImg.copyTo(mOutputImage(mvRects[4]));
			}
			if (isOutputTypeChanged(5)) {
				cv::Mat mMappingImg = GetOutputImage(5);
				mMappingImg.copyTo(mOutputImage(mvRects[5]));
			}*/
			
			///////save image
			//nMapImageID++;
			//if (nMapImageID % 20 == 0) {
			//	std::stringstream sss;
			//	sss << "../../bin/img/Map/" << nMapImageID << ".jpg";
			//	cv::imwrite(sss.str(), mOutputImage);
			//}
			///////save image

			imshow(strWindowName, mOutputImage);
			auto key = cv::waitKey(10);
			if (key == '1') {
				std::cout << "1" << std::endl;
			}
			if (key == '2') {
				std::cout << "2" << std::endl;
			}
		}
	}
	void Visualizer::SetBoolDoingProcess(bool b){
		std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
		mbDoingProcess = b;
	}
	bool Visualizer::isDoingProcess(){
		std::unique_lock<std::mutex> lockTemp(mMutexDoingProcess);
		return mbDoingProcess;
	}
	void Visualizer::ResizeImage(cv::Mat src, cv::Mat& dst) {
		cv::resize(src, dst, cv::Size(mnWidth / 2.0, mnHeight / 2.0));
	}
	void Visualizer::SetOutputImage(cv::Mat out, int type){
		std::unique_lock<std::mutex> lockTemp(mMutexOutput);
		mvOutputImgs[type] = out.clone();
		mvOutputChanged[type] = true;
	}
	cv::Mat Visualizer::GetOutputImage(int type){
		std::unique_lock<std::mutex> lockTemp(mMutexOutput);
		mvOutputChanged[type] = false;
		return mvOutputImgs[type].clone();
	}
	bool Visualizer::isOutputTypeChanged(int type){
		std::unique_lock<std::mutex> lockTemp(mMutexOutput);
		return mvOutputChanged[type];
	}

	void Visualizer::SetMap(Map* pMap){
		std::unique_lock<std::mutex> lockTemp(mMutexMap);
		mpMap = pMap;
	}
	Map* Visualizer::GetMap(){
		std::unique_lock<std::mutex> lockTemp(mMutexMap);
		return mpMap;
	}
	/*void Visualizer::AddUser(User* pUser) {
		std::unique_lock<std::mutex>lock(mMutexUserList);
		mspUserLists.insert(pUser);
	}
	void Visualizer::RemoveUser(User* pUser) {
		std::unique_lock<std::mutex>lock(mMutexUserList);
		if (mspUserLists.count(pUser))
			mspUserLists.erase(pUser);
	}
	std::vector<User*> Visualizer::GetUsers() {
		std::unique_lock<std::mutex>lock(mMutexUserList);
		return std::vector<User*>(mspUserLists.begin(), mspUserLists.end());
	}*/
}