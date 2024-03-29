#include <Segmentator.h>
#include <SLAM.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <User.h>
#include <MapPoint.h>
#include <Camera.h>
#include <Map.h>
#include <Visualizer.h>
#include <Plane.h>

namespace EdgeSLAM {

	int Segmentator::mnMaxObjectLabel;
	NewMapClass<int, Object*> Segmentator::ObjectPoints;
	std::vector<cv::Vec3b> Segmentator:: mvObjectLabelColors;
	std::set<MapPoint*> Segmentator::mspAllFloorPoints;
	std::set<MapPoint*> Segmentator::mspAllWallPoints;
	std::atomic<int> Segmentator::mnContentID = 0;
	
	Plane* Segmentator::floorPlane;
	Plane* Segmentator::wallPlane1;
	Plane* Segmentator::wallPlane2;

	cv::Mat Segmentator::Lw1;
	cv::Mat Segmentator::Lw2;

	Segmentator::Segmentator() {

	}
	Segmentator::~Segmentator() {

	}
	Object::Object():mnLabel(0), mnCount(0){
		matLabels = cv::Mat::zeros(Segmentator::mnMaxObjectLabel, 1, CV_16UC1);
	}
	Object::~Object(){
	
	}
	void Object::Update(int nLabel){
		
		std::unique_lock<std::mutex> lock(mMutexObject);
		matLabels.at<ushort>(nLabel)++;
		if (mnLabel == nLabel) {
			mnCount++;
		}
		else {
			int count = matLabels.at<ushort>(nLabel);
			if (count > mnCount) {
				double minVal;
				double maxVal;
				int minIdx, maxIdx;
				cv::minMaxIdx(matLabels, &minVal, &maxVal, &minIdx, &maxIdx, cv::Mat());
				mnLabel = maxIdx;
				mnCount = matLabels.at<ushort>(maxIdx);
			}
		}
	}
	int Object::GetLabel(){
		std::unique_lock<std::mutex> lock(mMutexObject);
		return mnLabel;
		/*cv::Mat label;
		{
			std::unique_lock<std::mutex> lock(mMutexObject);
			label = matLabels.clone();
		}
		double minVal;
		double maxVal;
		int minIdx, maxIdx;
		cv::minMaxIdx(matLabels, &minVal, &maxVal, &minIdx, &maxIdx, cv::Mat());
		return maxIdx;*/
	}
	cv::Mat Object::GetLabels(){
		std::unique_lock<std::mutex> lock(mMutexObject);
		return matLabels.clone();
	}
	int Object::GetCount(int l){
		std::unique_lock<std::mutex> lock(mMutexObject);
		return matLabels.at<ushort>(l);
		/*cv::Mat label;
		{
			std::unique_lock<std::mutex> lock(mMutexObject);
			label = matLabels.clone();
		}
		return label.at<ushort>(l);*/
	}
	void Segmentator::ProcessDevicePosition(SLAM* system, User* user, int id) {
		std::stringstream ss;
		ss << "/Load?keyword=DevicePosition" << "&id=" << id << "&src=" << user->userName;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();

		cv::Mat fdata = cv::Mat::zeros(4, 3, CV_32FC1);
		std::memcpy(fdata.data, res.data(), res.size());

		cv::Mat R = fdata.rowRange(0, 3).colRange(0, 3).clone();
		cv::Mat t = fdata.row(3).t();

		cv::Mat Rinv = R.t();
		cv::Mat pos = -Rinv*t;
		user->AddDevicePosition(pos);
	}

	cv::Point2f CalcLinePoint2(float val, cv::Mat mLine, bool opt) {
		float x, y;
		if (opt) {
			x = 0.0;
			y = val;
			if (mLine.at<float>(0) != 0)
				x = (-mLine.at<float>(2) - mLine.at<float>(1)*y) / mLine.at<float>(0);
		}
		else {
			y = 0.0;
			x = val;
			if (mLine.at<float>(1) != 0)
				y = (-mLine.at<float>(2) - mLine.at<float>(0)*x) / mLine.at<float>(1);
		}

		return cv::Point2f(x, y);
	}

	void Segmentator::ProcessContentGeneration(SLAM* system, User* user, int id) {

		auto map = system->GetMap(user->mapName);
		std::stringstream ss;
		ss << "/Load?keyword=ContentGeneration" << "&id=" << id << "&src=" << user->userName;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();

		cv::Mat fdata = cv::Mat::zeros(5, 3, CV_32FC1);
		std::memcpy(fdata.data, res.data(), res.size());
		
		cv::Mat R = fdata.rowRange(1, 4).colRange(0, 3).clone();
		cv::Mat t = fdata.row(4).t();
		
		cv::Mat Ximg = fdata.row(0).t();
		cv::Mat Rinv = R.t();
		cv::Mat tinv = -Rinv*t;
		cv::Mat Tinv = cv::Mat::eye(4, 4, CV_32FC1);
		Rinv.copyTo(Tinv.colRange(0, 3).rowRange(0, 3));
		tinv.copyTo(Tinv.col(3).rowRange(0,3));
		
		////click check
		/*float m1, m2;
		cv::Mat line1, line2;
		line1 = Segmentator::LineProjection(R, t, Segmentator::Lw1, user->mpCamera->Kfluker, m1);
		m1 = -line1.at<float>(0) / line1.at<float>(1);
		bool bSlopeOpt1 = abs(m1) > 1.0;
		float val1;
		if (bSlopeOpt1)
			val1 = 360.0;
		else
			val1 = 640.0;
		auto pt11 = CalcLinePoint2(0.0, line1, bSlopeOpt1);
		auto pt12 = CalcLinePoint2(val1, line1, bSlopeOpt1);

		line2 = Segmentator::LineProjection(R, t, Segmentator::Lw2, user->mpCamera->Kfluker, m2);
		m2 = -line2.at<float>(0) / line2.at<float>(1);
		bool bSlopeOpt2 = abs(m2) > 1.0;
		float val2;
		if (bSlopeOpt2)
			val2 = 360.0;
		else
			val2 = 640.0;
		auto pt21 = CalcLinePoint2(0.0, line2, bSlopeOpt2);
		auto pt22 = CalcLinePoint2(val2, line2, bSlopeOpt2);
		
		cv::Point2f sPt1, sPt2, ePt1, ePt2;
		if (pt11.x < pt12.x) {
			sPt1 = pt11;
			ePt1 = pt12;;
		}
		else {
			sPt1 = pt12;
			ePt1 = pt11;
		}
		if (pt21.x < pt22.x) {
			sPt2 = pt11;
			ePt1 = pt12;;
		}
		else {
			sPt1 = pt12;
			ePt1 = pt11;
		}*/
		////click check

		cv::Mat Kinv = user->GetCameraInverseMatrix();
		cv::Mat Pinv = PlaneProcessor::CalcInverPlaneParam(floorPlane->GetParam(), Tinv);
		
		cv::Mat Xnorm = Kinv*Ximg;
		float depth = PlaneProcessor::CalculateDepth(Xnorm, Pinv);
		cv::Mat Xw = PlaneProcessor::CreateWorldPoint(Xnorm, Tinv, depth);
		std::cout << Xw.t() << std::endl;
		////Store Content
		ss.str("");
		ss << "/Store?keyword=Content&id=" << ++mnContentID << "&src=ContentServer&type2=" << user->userName<<"&id2="<<id;
		res = mpAPI->Send(ss.str(), Xw.data, Xw.rows * sizeof(float));
		
	}

	cv::Mat Segmentator::CalcFlukerLine(cv::Mat P1, cv::Mat P2) {
		cv::Mat PLw1, Lw1, NLw1;
		PLw1 = P1*P2.t() - P2*P1.t();
		Lw1 = cv::Mat::zeros(6, 1, CV_32FC1);
		Lw1.at<float>(3) = PLw1.at<float>(2, 1);
		Lw1.at<float>(4) = PLw1.at<float>(0, 2);
		Lw1.at<float>(5) = PLw1.at<float>(1, 0);
		NLw1 = PLw1.col(3).rowRange(0, 3);
		NLw1.copyTo(Lw1.rowRange(0, 3));

		return Lw1;
	}
	cv::Mat Segmentator::LineProjection(cv::Mat R, cv::Mat t, cv::Mat Lw1, cv::Mat K, float& m) {
		cv::Mat T2 = cv::Mat::zeros(6, 6, CV_32FC1);
		R.copyTo(T2.rowRange(0, 3).colRange(0, 3));
		R.copyTo(T2.rowRange(3, 6).colRange(3, 6));
		cv::Mat tempSkew = cv::Mat::zeros(3, 3, CV_32FC1);
		tempSkew.at<float>(0, 1) = -t.at<float>(2);
		tempSkew.at<float>(1, 0) = t.at<float>(2);
		tempSkew.at<float>(0, 2) = t.at<float>(1);
		tempSkew.at<float>(2, 0) = -t.at<float>(1);
		tempSkew.at<float>(1, 2) = -t.at<float>(0);
		tempSkew.at<float>(2, 1) = t.at<float>(0);
		tempSkew *= R;
		tempSkew.copyTo(T2.rowRange(0, 3).colRange(3, 6));
		cv::Mat Lc = T2*Lw1;
		cv::Mat Nc = Lc.rowRange(0, 3);
		cv::Mat res = K*Nc;
		/*if (res.at<float>(0) < 0)
			res *= -1;
		if (res.at<float>(0) != 0)
			m = res.at<float>(1) / res.at<float>(0);
		else
			m = 9999.0;*/
		return res.clone();
	}

	void Segmentator::ProcessPlanarModeling(SLAM* system, User* user) {
		auto map = system->GetMap(user->mapName);
		//map->GetAllKeyFrames();
		//std::cout << "??????" << std::endl;
		 std::cout << "Ploor test = " << mspAllFloorPoints.size() << std::endl;
		 floorPlane = new Plane();
		 wallPlane1 = new Plane();
		 wallPlane2 = new Plane();
		 map->ClearPlanarMPs();
		 {
			 std::vector<MapPoint*> vpMPs(mspAllFloorPoints.begin(), mspAllFloorPoints.end());
			 std::vector<MapPoint*> vpOutlierMPs;
			 floorPlane->mbInit = PlaneProcessor::PlaneInitialization(floorPlane, vpMPs, vpOutlierMPs);

			 if (floorPlane->mbInit) {
				 for (int i = 0, iend = floorPlane->mvpMPs.size(); i < iend; i++) {
					 auto pMP = floorPlane->mvpMPs[i];
					 if (!pMP || pMP->isBad())
						 continue;
					 map->AddPlanarMP(pMP->GetWorldPos(), 0);
				 }
			 }
		 }
		 std::vector<MapPoint*> vpOutlierWallMPs, vpOutlierWallMPs2;
		 {
			 std::vector<MapPoint*> vpMPs(mspAllWallPoints.begin(), mspAllWallPoints.end());
			 
			 wallPlane1->mbInit = PlaneProcessor::PlaneInitialization(wallPlane1, vpMPs, vpOutlierWallMPs, 1500, 0.01);
			 if (wallPlane1->mbInit) {
				 for (int i = 0, iend = wallPlane1->mvpMPs.size(); i < iend; i++) {
					 auto pMP = wallPlane1->mvpMPs[i];
					 if (!pMP || pMP->isBad())
						 continue;
					 map->AddPlanarMP(pMP->GetWorldPos(), 1);
				 }
			 }
		 }
		 if (wallPlane1->mbInit) {
			 wallPlane2->mbInit = PlaneProcessor::PlaneInitialization(wallPlane2, vpOutlierWallMPs, vpOutlierWallMPs2, 1500, 0.01);
			 //std::cout << vpOutlierWallMPs.size() << " " << vpOutlierWallMPs2.size() << std::endl;
			 if (wallPlane2->mbInit) {
				 
				 for (int i = 0, iend = wallPlane2->mvpMPs.size(); i < iend; i++) {
					 auto pMP = wallPlane2->mvpMPs[i];
					 if (!pMP || pMP->isBad())
						 continue;
					 map->AddPlanarMP(pMP->GetWorldPos(), 2);
				 }
			 }
		 }
		 if (floorPlane->mbInit && wallPlane1->mbInit) {
			 Lw1 = CalcFlukerLine(floorPlane->GetParam(), wallPlane1->GetParam());
		 }
		 if (floorPlane->mbInit && wallPlane2->mbInit) {
			 Lw2 = CalcFlukerLine(floorPlane->GetParam(), wallPlane2->GetParam());
		 }
		 std::cout << "plane res = " << floorPlane->mvpMPs.size() << ", " << wallPlane1->mvpMPs.size()<<" "<< wallPlane2->mvpMPs.size() << std::endl;
	}
	void Segmentator::ProcessSegmentation(ThreadPool::ThreadPool* pool, SLAM* system, std::string user, int id)
	{
		auto pUser = system->GetUser(user);
		std::stringstream ss;
		ss << "/Load?keyword=Segmentation"  << "&id=" << id<< "&src=" << user;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();
		
		int w = pUser->mpCamera->mnWidth;
		int h = pUser->mpCamera->mnHeight;
		cv::Mat labeled = cv::Mat::zeros(h, w, CV_8UC1);
		std::memcpy(labeled.data, res.data(), res.size());

		////segmentation image
		cv::Mat segcolor = cv::Mat::zeros(h,w, CV_8UC3);
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int label = labeled.at<uchar>(y, x) + 1;

				switch (label) {
				case (int)ObjectLabel::FLOOR:
					segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
					//mask_floor.at<uchar>(y, x) = 255;
					break;
				case (int)ObjectLabel::WALL:
					segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
					//mask_wall.at<uchar>(y, x) = 255;
					break;
				case (int)ObjectLabel::CEIL:
					segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
					//mask_ceil.at<uchar>(y, x) = 255;
					break;
				default:
					segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
					break;
				}
			}
		}

		auto F = pUser->mapFrames[id];
		for (int i = 0; i < F->N; i++) {
			auto pMP = F->mvpMapPoints[i];
			if (!pMP || pMP->isBad())
				continue;
			auto pt = F->mvKeys[i].pt;
			int label = labeled.at<uchar>(pt.y, pt.x) + 1;

			////update object label
			int nmpid = pMP->mnId;
			Object* obj = nullptr;
			if (ObjectPoints.Count(nmpid)) {
				obj = ObjectPoints.Get(nmpid);
			}
			else {
				obj = new Object();
				ObjectPoints.Update(nmpid, obj);
			}
			obj->Update(label);
			////update object label

			////detect structure poitns
			switch (label) {
			case (int)ObjectLabel::FLOOR:
				if (mspAllFloorPoints.count(pMP))
					continue;
				mspAllFloorPoints.insert(pMP);
				break;
			case (int)ObjectLabel::WALL:
				if (mspAllWallPoints.count(pMP))
					continue;
				mspAllWallPoints.insert(pMP);
				break;
			case (int)ObjectLabel::CEIL:
				break;
			}
			////detect structure poitns
		}

		/////save image
		std::stringstream sss;
		sss << "../../bin/img/" << pUser->userName << "/Segmentation/" << id << ".jpg";
		cv::imwrite(sss.str(), segcolor);
		/////save image

		//auto KF = pUser->mapKeyFrames[F->mnKeyFrameId];
		/*cv::Mat resized_test;
		cv::resize(segcolor, resized_test, cv::Size(segcolor.cols / 2, segcolor.rows / 2));*/
		system->mpVisualizer->ResizeImage(segcolor, segcolor);
		system->mpVisualizer->SetOutputImage(segcolor, 1);

		////save image
		/*std::stringstream ssa;
		ssa << "../../bin/SLAM/Images/img_" << F->mnFrameID << "_seg.jpg";
		cv::imwrite(ssa.str(), segcolor);

		ssa.str("");
		ssa << "../../bin/SLAM/Images/img_" << F->mnFrameID << "_ori.jpg";
		cv::imwrite(ssa.str(), F->imgColor);*/
		////save image
	}
	void Segmentator::ProcessDepthEstimation(ThreadPool::ThreadPool* pool, SLAM* system, std::string user, int id)
	{
		auto pUser = system->GetUser(user);
		std::stringstream ss;
		ss << "/Load?keyword=DepthEstimation" << "&id=" << id << "&src=" << user;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();

		int w = pUser->mpCamera->mnWidth;
		int h = pUser->mpCamera->mnHeight;
		cv::Mat depthImg = cv::Mat::zeros(h, w, CV_32FC1);
		std::memcpy(depthImg.data, res.data(), res.size());

		{
			cv::Mat depth;
			cv::normalize(depthImg, depth, 0, 255, cv::NORM_MINMAX, CV_32FC1);
			
			/////save image
			std::stringstream sss;
			sss << "../../bin/img/" << pUser->userName << "/Depth/" << id << ".jpg";
			cv::imwrite(sss.str(), depth);
			/////save image
		}

		////segmentation image
		//cv::Mat depthImg = cv::Mat::zeros(h, w, CV_8UC3);
		//for (int y = 0; y < h; y++) {
		//	for (int x = 0; x < w; x++) {
		//		int label = labeled.at<uchar>(y, x) + 1;

		//		switch (label) {
		//		case (int)ObjectLabel::FLOOR:
		//			segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
		//			//mask_floor.at<uchar>(y, x) = 255;
		//			break;
		//		case (int)ObjectLabel::WALL:
		//			segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
		//			//mask_wall.at<uchar>(y, x) = 255;
		//			break;
		//		case (int)ObjectLabel::CEIL:RequestDepth
		//			segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
		//			//mask_ceil.at<uchar>(y, x) = 255;
		//			break;
		//		}
		//	}
		//}
		
		//////////////////
		auto F = pUser->mapFrames[id];
		auto map = system->GetMap(pUser->mapName);
		std::vector<std::tuple<cv::Point2f, float, int>> vecTuples;

		cv::Mat R, t;
		R = F->GetRotation();
		t = F->GetTranslation();

		////depth 정보 저장 및 포인트와 웨이트 정보를 튜플로 저장
		cv::Mat Rcw2 = R.row(2);
		Rcw2 = Rcw2.t();
		float zcw = t.at<float>(2);
		auto vpMPs = F->mvpMapPoints;
		for (size_t i = 0, iend = F->N; i < iend; i++) {
			auto pMPi = vpMPs[i];
			if (!pMPi || pMPi->isBad())
				continue;
			auto pt = F->mvKeysUn[i].pt;
			cv::Mat x3Dw = pMPi->GetWorldPos();
			float z = (float)Rcw2.dot(x3Dw) + zcw;
			std::tuple<cv::Point2f, float, int> data = std::make_tuple(std::move(pt), 1.0 / z, pMPi->Observations());//cv::Point2f(pt.x / 2, pt.y / 2)
			vecTuples.push_back(data);
		}
		////웨이트와 포인트 정보로 정렬
		std::sort(vecTuples.begin(), vecTuples.end(),
			[](std::tuple<cv::Point2f, float, int> const &t1, std::tuple<cv::Point2f, float, int> const &t2) {
			if (std::get<2>(t1) == std::get<2>(t2)) {
				return std::get<0>(t1).x != std::get<0>(t2).x ? std::get<0>(t1).x > std::get<0>(t2).x : std::get<0>(t1).y > std::get<0>(t2).y;
			}
			else {
				return std::get<2>(t1) > std::get<2>(t2);
			}
		}
		);
		////파라메터 검색 및 뎁스 정보 복원
		int nTotal = 40;
		if (vecTuples.size() > nTotal) {
			int nData = vecTuples.size();//nTotal;
			cv::Mat A = cv::Mat::ones(nData, 2, CV_32FC1);
			cv::Mat B = cv::Mat::zeros(nData, 1, CV_32FC1);

			for (size_t i = 0; i < nData; i++) {
				auto data = vecTuples[i];
				auto pt = std::get<0>(data);
				auto invdepth = std::get<1>(data);
				auto nConnected = std::get<2>(data);

				float p = depthImg.at<float>(pt);
				A.at<float>(i, 0) = invdepth;
				B.at<float>(i) = p;
			}

			cv::Mat X = A.inv(cv::DECOMP_QR)*B;
			float a = X.at<float>(0);
			float b = X.at<float>(1);

			depthImg = (depthImg - b) / a;
			for (int x = 0, cols = depthImg.cols; x < cols; x++) {
				for (int y = 0, rows = depthImg.rows; y < rows; y++) {
					float val = 1.0 / depthImg.at<float>(y, x);
					/*if (val < 0.0001)
					val = 0.5;*/
					depthImg.at<float>(y, x) = val;
				}
			}
			//////복원 확인
			//cv::Mat invK = F->InvK.clone();
			//cv::Mat Rinv, Tinv;
			//cv::Mat Pinv = F->GetPoseInverse();
			//Rinv = Pinv.colRange(0, 3).rowRange(0, 3);
			//Tinv = Pinv.col(3).rowRange(0, 3);
			//map->ClearDepthMPs();
			//int inc = 5;
			//for (size_t row = inc, rows = depthImg.rows; row < rows; row+= inc) {
			//	for (size_t col = inc, cols = depthImg.cols; col < cols; col+= inc) {
			//		cv::Point2f pt(col, row);
			//		float depth = depthImg.at<float>(pt);
			//		if (depth < 0.0001)
			//			continue;
			//		cv::Mat a = Rinv*(invK*(cv::Mat_<float>(3, 1) << pt.x, pt.y, 1.0)*depth) + Tinv;
			//		map->AddDepthMP(a);
			//	}
			//}
			//////복원 확인
		}
		//////////////////
		
		cv::Mat depth, resized_test;
		cv::normalize(depthImg, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::cvtColor(depth, depth, cv::COLOR_GRAY2BGR);

		///////save image
		//std::stringstream sss;
		//sss << "../../bin/img/" << pUser->userName << "/Depth/" << id << ".jpg";
		//cv::imwrite(sss.str(), depth);
		///////save image

		//cv::resize(depth, resized_test, cv::Size(depth.cols / 2, depth.rows / 2));
		system->mpVisualizer->ResizeImage(depth, resized_test);
		system->mpVisualizer->SetOutputImage(resized_test, 2);

		////save image
		/*std::stringstream ssa;
		ssa << "../../bin/SLAM/Images/img_" << F->mnFrameID << "_depth.jpg";
		cv::imwrite(ssa.str(), depth);*/
		////save image

		/*cv::Mat resized_test;
		cv::resize(segcolor, resized_test, cv::Size(segcolor.cols / 2, segcolor.rows / 2));
		system->mpVisualizer->SetOutputImage(resized_test, 2);*/
	}
	
	void Segmentator::RequestSegmentation(std::string user, int id)
	{
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		std::stringstream ss;
		ss << "/Store?keyword=RequestDepth&id=" << id << "&src=" << user;
		auto res = mpAPI->Send(ss.str(), "");
		ss.str("");
		ss << "/Store?keyword=RequestSegmentation&id=" << id <<"&src="<<user;
		res = mpAPI->Send(ss.str(), "");
	}
	void Segmentator::Init() {
		cv::Mat colormap = cv::Mat::zeros(256, 3, CV_8UC1);
		cv::Mat ind = cv::Mat::zeros(256, 1, CV_8UC1);
		for (int i = 1; i < ind.rows; i++) {
			ind.at<uchar>(i) = i;
		}

		for (int i = 7; i >= 0; i--) {
			for (int j = 0; j < 3; j++) {
				cv::Mat tempCol = colormap.col(j);
				int a = pow(2, j);
				int b = pow(2, i);
				cv::Mat temp = ((ind / a) & 1) * b;
				tempCol |= temp;
				tempCol.copyTo(colormap.col(j));
			}
			ind /= 8;
		}

		for (int i = 0; i < colormap.rows; i++) {
			cv::Vec3b color = cv::Vec3b(colormap.at<uchar>(i, 0), colormap.at<uchar>(i, 1), colormap.at<uchar>(i, 2));
			mvObjectLabelColors.push_back(color);
		}
		mnMaxObjectLabel = mvObjectLabelColors.size();
	}
}
