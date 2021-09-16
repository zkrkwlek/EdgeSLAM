#include <Visualizer.h>
#include <SLAM.h>
#include <Map.h>
#include <MapPoint.h>
#include <User.h>
#include <Segmentator.h>

namespace EdgeSLAM {
	Visualizer::Visualizer() {

	}
	Visualizer::Visualizer(SLAM* pSystem):mpSystem(pSystem), mpMap(nullptr), mnVisScale(20), mnDisplayX(100), mnDisplayY(100), mbDoingProcess(false){

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
		int* tempData = (int*)userdata;

		if (event == cv::EVENT_LBUTTONDOWN)
		{
			//std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
			tempData[2] = x - tempData[0];
			tempData[3] = y;

			////button interface
			if (tempData[2] < 50 && y < 50) {
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
			tempData[4] = x - tempData[0];
			tempData[5] = y;
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
			//std::cout << "Wheel event detection" << std::endl;
			if (flags > 0) {
				//scroll up
				tempData[1] += 20;
			}
			else {
				//scroll down
				tempData[1] -= 20;
				if (tempData[1] <= 0) {
					tempData[1] = 20;
				}
			}
		}
	}

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

		//4
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
		int nDisCols = leftImg1.cols + mapImage.cols;// +kfWindowImg.cols;
		mOutputImage = cv::Mat::zeros(nDisRows, nDisCols, CV_8UC3);
	}

	
	void Visualizer::Run(){
		SetAxisMode();
		int nMapImageID = 0;
		
		//현재 여기서 시각화가 안됨
		//원인 찾는 중
		cv::imshow("Output::Display", mOutputImage);
		cv::moveWindow("Output::Display", mnDisplayX, mnDisplayY);
		int mapControlData[8] = { 0, };
		mapControlData[0] = mnWidth;
		mapControlData[1] = mnVisScale;
		cv::setMouseCallback("Output::Display", EdgeSLAM::Visualizer::CallBackFunc, (void*)mapControlData);

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

			cv::Mat tempVis = mVisPoseGraph.clone();
			auto pMap = GetMap();
			if (pMap) {

				auto mmpMap = pMap->GetAllMapPoints();
				for (auto iter = mmpMap.begin(); iter != mmpMap.end(); iter++) {
					auto pMPi = *iter;// ->first;
					if (!pMPi || pMPi->isBad())
						continue;

					cv::Scalar color = cv::Scalar(0, 0, 0);
					if (Segmentator::ObjectPoints.count(pMPi->mnId)) {
						auto obj = Segmentator::ObjectPoints[pMPi->mnId];
						if (obj) {
							int label = obj->GetLabel();
							color = Segmentator::mvObjectLabelColors[label];
						}
					}
					
					cv::Mat x3D = pMPi->GetWorldPos();
					cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
					tpt += mVisMidPt;
					
					cv::circle(tempVis, tpt, 2, color, -1);
				}

				{
					
					
					for (int i = 0, iend = 3; i < iend; i++) {
						auto vMPs = pMap->GetPlanarMPs(i);
						for (int j = 0; j < vMPs.size(); j++) {
							cv::Mat x3D = vMPs[j];
							cv::Point2f tpt = cv::Point2f(x3D.at<float>(mnAxis1) * mnVisScale, x3D.at<float>(mnAxis2) * mnVisScale);
							tpt += mVisMidPt;
							cv::circle(tempVis, tpt, 4, planeColors[i], -1);
						}
					}
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

					//{
					//	////gyro testa
					//	cv::Mat T = user->GetPose();
					//	cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
					//	auto pos = user->GetPosition();
					//	cv::Point2f pt1 = cv::Point2f(pos.at<float>(mnAxis1)* mnVisScale, pos.at<float>(mnAxis2)* mnVisScale);
					//	pt1 += mVisMidPt;
					//	cv::circle(tempVis, pt1, 3, cv::Scalar(0, 0, 255), -1);

					//	cv::Mat directionZ = R.row(2);
					//	cv::Point2f dirPtZ = cv::Point2f(directionZ.at<float>(mnAxis1)* mnVisScale / 10.0, directionZ.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
					//	cv::line(tempVis, pt1, dirPtZ, cv::Scalar(255, 0, 0), 2);

					//	cv::Mat directionY = R.row(1);
					//	cv::Point2f dirPtY = cv::Point2f(directionY.at<float>(mnAxis1)* mnVisScale / 10.0, directionY.at<float>(mnAxis2)* mnVisScale / 10.0) + pt1;
					//	cv::line(tempVis, pt1, dirPtY, cv::Scalar(0, 255, 0), 2);

					//	cv::Mat directionX = R.row(0);
					//	cv::Point2f dirPtX1 = pt1 + cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
					//	cv::Point2f dirPtX2 = pt1 - cv::Point2f(directionX.at<float>(mnAxis1)* mnVisScale / 10.0, directionX.at<float>(mnAxis2)* mnVisScale / 10.0);
					//	cv::line(tempVis, dirPtX1, dirPtX2, cv::Scalar(0, 0, 255), 2);

					//}

					auto pos = user->GetPosition();
					cv::Point2f pt1 = cv::Point2f(pos.at<float>(mnAxis1)* mnVisScale, pos.at<float>(mnAxis2)* mnVisScale);
					pt1 += mVisMidPt;
					if(user->mbMapping)
						cv::circle(tempVis, pt1, 4, cv::Scalar(0, 0, 255), -1);
					else {
						cv::circle(tempVis, pt1, 4, userColors[i], -1);
					}
					auto dPoses = user->GetDevicePositions();
					for (int j = 0, jend = dPoses.size(); j < jend; j++) {
						auto pos = dPoses[j];
						cv::Point2f pt1 = cv::Point2f(pos.at<float>(mnAxis1)* mnVisScale, pos.at<float>(mnAxis2)* mnVisScale);
						pt1 += mVisMidPt;
						cv::circle(tempVis, pt1, 2, userColors[i], -1);
					}
				}
			}

			SetOutputImage(tempVis, 4);
			////////Update Map Visualizer
			if (isOutputTypeChanged(0)) {
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
			}
			
			///////save image
			//nMapImageID++;
			//if (nMapImageID % 20 == 0) {
			//	std::stringstream sss;
			//	sss << "../../bin/img/Map/" << nMapImageID << ".jpg";
			//	cv::imwrite(sss.str(), mOutputImage);
			//}
			///////save image

			imshow("Output::Display", mOutputImage);
			cv::waitKey(1);
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