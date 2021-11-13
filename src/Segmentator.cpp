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

#include <FeatureDetector.h>
#include <SearchPoints.h>
#include <Converter.h>
#include <Utils.h>

namespace EdgeSLAM {
	std::string Segmentator::strLabel = "wall,building,sky,floor,tree,ceiling,road,bed,windowpane,grass,cabinet,sidewalk,person,earth,door,table,mountain,plant,curtain,chair,car,water,painting,sofa,shelf,house,sea,mirror,rug,field,armchair,seat,fence,desk,rock,wardrobe,lamp,bathtub,railing,cushion,base,box,column,signboard,chest of drawers,counter,sand,sink,skyscraper,fireplace,refrigerator,grandstand,path,stairs,runway,case,pool table,pillow,screen door,stairway,river,bridge,bookcase,blind,coffee table,toilet,flower,book,hill,bench,countertop,stove,palm,kitchen island,computer,swivel chair,boat,bar,arcade machine,hovel,bus,towel,light,truck,tower,chandelier,awning,streetlight,booth,television,airplane,dirt track,apparel,pole,land,bannister,escalator,ottoman,bottle,buffet,poster,stage,van,ship,fountain,conveyer belt,canopy,washer,plaything,swimming pool,stool,barrel,basket,waterfall,tent,bag,minibike,cradle,oven,ball,food,step,tank,trade name,microwave,pot,animal,bicycle,lake,dishwasher,screen,blanket,sculpture,hood,sconce,vase,traffic light,tray,ashcan,fan,pier,crt screen,plate,monitor,bulletin board,shower,radiator,glass,clock,flag";
	std::string Segmentator::strYoloObjectLabel = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush";
	std::vector<std::string> Segmentator::mvStrLabels;
	std::vector<std::string> Segmentator::mvStrObjectLabels;
	int Segmentator::mnMaxObjectLabel;
	NewMapClass<int, cv::Mat> Segmentator::SegmentedFrames;
	NewMapClass<int, Object*> Segmentator::ObjectPoints;
	std::vector<cv::Vec3b> Segmentator:: mvObjectLabelColors;
	std::set<MapPoint*> Segmentator::mspAllFloorPoints;
	std::set<MapPoint*> Segmentator::mspAllWallPoints;
	std::atomic<int> Segmentator::mnContentID = 0;
	
	Plane* Segmentator::floorPlane = nullptr;
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

		////ÀÓ½Ã·Î
		if (!floorPlane)
			return;

		cv::Mat Kinv = user->GetCameraInverseMatrix();
		cv::Mat Pinv = PlaneProcessor::CalcInverPlaneParam(floorPlane->GetParam(), Tinv);
		
		cv::Mat Xnorm = Kinv*Ximg;
		float depth = PlaneProcessor::CalculateDepth(Xnorm, Pinv);
		cv::Mat Xw = PlaneProcessor::CreateWorldPoint(Xnorm, Tinv, depth);
		std::cout << Xw.t() << std::endl;

		cv::Mat data = cv::Mat::ones(400, 1, CV_32FC1);
		data.at<float>(0) = Xw.at<float>(0);
		data.at<float>(1) = Xw.at<float>(1);
		data.at<float>(2) = Xw.at<float>(2);

		////Store Content
		ss.str("");
		ss << "/Store?keyword=Content&id=" << ++mnContentID << "&src=ContentServer&type2=" << user->userName;//<< "&id2=" << id;
		res = mpAPI->Send(ss.str(), data.data, data.rows * sizeof(float));
		
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

	bool Segmentator::ConnectedComponentLabeling(cv::Mat src, cv::Mat& dst, cv::Mat& stat, std::string strLabel){
		//dst = src.clone();
		cv::Mat img_labels, stats, centroids;
		int numOfLables = connectedComponentsWithStats(src, img_labels, stats, centroids, 8, CV_32S);

		if (numOfLables == 0)
			return false;

		int maxArea = 0;
		int maxIdx = 0;
		//¶óº§¸µ µÈ ÀÌ¹ÌÁö¿¡ °¢°¢ Á÷»ç°¢ÇüÀ¸·Î µÑ·¯½Î±â 
		for (int j = 0; j < numOfLables; j++) {
			int area = stats.at<int>(j, cv::CC_STAT_AREA);
			if (area > maxArea) {
				maxArea = area;
				maxIdx = j;
			}
		}
		/*int left = stats.at<int>(maxIdx, CC_STAT_LEFT);
		int top = stats.at<int>(maxIdx, CC_STAT_TOP);
		int width = stats.at<int>(maxIdx, CC_STAT_WIDTH);
		int height = stats.at<int>(maxIdx, CC_STAT_HEIGHT);*/

		for (int j = 0; j < numOfLables; j++) {
			if (j == maxIdx)
				continue;
			int area = stats.at<int>(j, cv::CC_STAT_AREA);
			if (area < 400)
				continue;

			int left = stats.at<int>(j, cv::CC_STAT_LEFT);
			int top = stats.at<int>(j, cv::CC_STAT_TOP);
			int width = stats.at<int>(j, cv::CC_STAT_WIDTH);
			int height = stats.at<int>(j, cv::CC_STAT_HEIGHT);
			rectangle(dst, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(255, 255, 255));
			cv::putText(dst, strLabel, cv::Point(left, top-6), 1, 1.5, cv::Scalar::all(255));
		}
		stat = stats.row(maxIdx).clone();
		return true;
	}

	int prevID = -1;
	void Segmentator::ProcessSegmentation(ThreadPool::ThreadPool* pool, SLAM* system, std::string user, int id)
	{
		auto pUser = system->GetUser(user);
		std::stringstream ss;
		ss << "/Load?keyword=Segmentation"  << "&id=" << id<< "&src=" << user;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();

		std::cout << "seg = " << n2 << std::endl;

		//ï¿½ï¿½ï¿½ï¿½
		cv::Mat temp = cv::Mat::zeros(n2, 1, CV_8UC1);
		std::memcpy(temp.data, res.data(), res.size());
		cv::Mat labeled = cv::imdecode(temp, cv::IMREAD_GRAYSCALE);

		int w = labeled.cols;
		int h = labeled.rows;

		int oriw = pUser->mpCamera->mnWidth;
		int orih = pUser->mpCamera->mnHeight;

		float sw = ((float)w) / oriw;
		float sh = ((float)h) / orih;

		std::cout << w << " " << h << std::endl;

		/*
		//ï¿½ï¿½ï¿½ï¿½ ï¿½Æ´ï¿½
		
		cv::Mat labeled = cv::Mat::zeros(h, w, CV_8UC1);
		std::memcpy(labeled.data, res.data(), res.size());*/

		////segmentation image
		/*std::vector<cv::Mat> vecBinaryLabelImages(mnMaxObjectLabel);
		cv::Mat temp = cv::Mat::zeros(h, w, CV_8UC1);
		for (int i = 0, iend = mnMaxObjectLabel; i < iend; i++) {
			 vecBinaryLabelImages[i] = temp.clone();
		}*/
		
		cv::Mat segcolor = cv::Mat::zeros(h,w, CV_8UC3);
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				int label = labeled.at<uchar>(y, x) + 1;
				segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
				//vecBinaryLabelImages[label].at<uchar>(y, x) = 255;

				//switch (label) {
				//case (int)ObjectLabel::FLOOR:
				//	segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
				//	//mask_floor.at<uchar>(y, x) = 255;
				//	break;
				//case (int)ObjectLabel::WALL:
				//	segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
				//	//mask_wall.at<uchar>(y, x) = 255;
				//	break;
				//case (int)ObjectLabel::CEIL:
				//	segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
				//	//mask_ceil.at<uchar>(y, x) = 255;
				//	break;
				//default:
				//	segcolor.at<cv::Vec3b>(y, x) = Segmentator::mvObjectLabelColors[label];
				//	break;
				//}
			}
		}
		////add labeled image
		SegmentedFrames.Update(id, labeled);
		
		////test matching object points and frame
		//if (prevID > 0) {
		//	auto F1 = pUser->mapFrames[id];
		//	auto F2 = pUser->mapFrames[prevID];
		//	std::vector<std::pair<int, int>> matches;
		//	std::vector<cv::Point2f> pts;
		//	
		//	DBoW3::BowVector mBowVec;
		//	DBoW3::FeatureVector mFeatVec;

		//	cv::Mat matObj = cv::Mat::zeros(0, F1->mDescriptors.cols, F1->mDescriptors.type());
		//	for (int i = 0; i < F1->N; i++) {
		//		auto pt = F1->mvKeys[i].pt;
		//		int label = labeled.at<uchar>(pt) + 1;
		//		if (label == (int)ObjectLabel::FLOOR) {
		//			pts.push_back(pt);
		//			matObj.push_back(F1->mDescriptors.row(i));
		//			
		//		}
		//	}
		//	std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(matObj);
		//	F1->mpVoc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);  // 5 is better
		//	
		//	SearchPoints::SearchObject(matObj, mFeatVec, F2, matches,50.0,50.0,0.7,false);
		//	cv::Mat m1 = F1->imgColor.clone();
		//	cv::Mat m2 = F2->imgColor.clone();
		//	for (int i = 0, iend = matches.size(); i < iend; i++) {
		//		int idx1 = matches[i].first;
		//		int idx2 = matches[i].second;
		//		cv::circle(m1, pts[idx1], 3, cv::Scalar(255, 0, 255), -1);
		//		cv::circle(m2, F2->mvKeys[idx2].pt, 3, cv::Scalar(255, 0, 255), -1);
		//	}

		//	imshow("obj", m1);
		//	imshow("obj2", m2);
		//	cv::waitKey(1);
		//	//std::cout << "aaaa = " << matches.size() << std::endl;
		//}
		//prevID = id;
		////test matching object points and frame

		////Update Object Points && 
		//int nLabel = mvStrLabels.size();
		//cv::Mat count = cv::Mat::zeros(nLabel, 1, CV_32SC1);

		auto F = pUser->mapFrames[id];

		for (int i = 0; i < F->N; i++) {
			auto pMP = F->mvpMapPoints[i];
			if (!pMP || pMP->isBad())
				continue;
			auto pt = F->mvKeys[i].pt;
			pt.x *= sw;
			pt.y *= sh;
			int label = labeled.at<uchar>(pt.y, pt.x)+1;
			//count.at<int>(label)++;

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
			
			//////detect structure poitns
			//label++;
			//switch (label) {
			//case (int)ObjectLabel::FLOOR:
			//	if (mspAllFloorPoints.count(pMP))
			//		continue;
			//	mspAllFloorPoints.insert(pMP);
			//	break;
			//case (int)ObjectLabel::WALL:
			//	if (mspAllWallPoints.count(pMP))
			//		continue;
			//	mspAllWallPoints.insert(pMP);
			//	break;
			//case (int)ObjectLabel::CEIL:
			//	break;
			//}
			//////detect structure poitns
		}
		
		////Connected Component Labeling
		////put text
		//cv::Mat objLabel = count > 20; //thresh·Î »©±â
		//for (int i = 0, iend = mvStrLabels.size(); i < iend; i++) {
		//	int nLabel = i + 1;
		//	switch (nLabel) {
		//	case (int)ObjectLabel::FLOOR:
		//	case (int)ObjectLabel::WALL:
		//	case (int)ObjectLabel::CEIL:
		//		continue;
		//	}
		//	if (!objLabel.at<uchar>(i)) {
		//		continue;
		//	}
		//	cv::Mat stat;
		//	ConnectedComponentLabeling(vecBinaryLabelImages[i], segcolor, stat, mvStrLabels[i]);
		//}
		////Connected Component Labeling
		////put text
		{
			////////
			//////¿ÀºêÁ§Æ® Á¤º¸ Ãâ·ÂÇÏ±â
			//{
			//	std::cout <<"Object label = "<< nLabel;
			//	std::stringstream ss;
			//	ss << "../bin/img/" << pUser->userName << "/obj.csv";
			//	std::ofstream output(ss.str(), std::ios::app);
			//	output << id << " ";
			//	for (int i = 0; i < nLabel; i++) {
			//		int val = count.at<int>(i);
			//		output << val;
			//		if (i == nLabel - 1)
			//			output << std::endl;
			//		else
			//			output << " ";
			//	}
			//	output.close();
			//}
			//cv::Mat vis = cv::Mat::zeros(500, nLabel*5, CV_8UC3);
			//for (int i = 0; i < nLabel; i++) {
			//	int val = 500 - count.at<int>(i);
			//	cv::Point2f pt1(i*5, 500);
			//	cv::Point2f pt2(pt1.x+5, val);
			//	cv::rectangle(vis, pt1, pt2, cv::Scalar(255,0,0), -1);
			//}
			//cv::imshow("obj test hist", vis); cv::waitKey(1);
			//////Update Object Points

			///////save image
			//std::stringstream sss;
			//sss << "../bin/img/" << pUser->userName << "/" << id << "_seg.jpg";
			//cv::imwrite(sss.str(), segcolor);
			///////save image
		}

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
		ssa << "../../bin/SLAM/Images/img_" << F->mnFrameID << "_ ori.jpg";
		cv::imwrite(ssa.str(), F->imgColor);*/
		////save image
	}
	void Segmentator::ProcessObjectDetection(ThreadPool::ThreadPool* pool, SLAM* system, std::string user, int id){
		auto pUser = system->GetUser(user);
		std::stringstream ss;
		ss << "/Load?keyword=ObjectDetection" << "&id=" << id << "&src=" << user;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();
		int n = n2 / 24;
		cv::Mat data = cv::Mat::zeros(n, 6, CV_32FC1);
		
		auto F = pUser->mapFrames[id];
		cv::Mat dst = F->imgColor.clone();
		std::memcpy(data.data, res.data(), res.size());

		cv::Mat objImg = F->imgGray.clone();//(maxRect)
		
		std::chrono::high_resolution_clock::time_point s1 = std::chrono::high_resolution_clock::now();

		auto vpMPs = F->mvpMapPoints;

		cv::Ptr<cv::Feature2D> detector = cv::ORB::create(2000);
		for (int j = 0; j < n; j++) {
			int label = (int)data.at<float>(j, 0);
			float conf = data.at<float>(j, 1);
			std::stringstream ss;
			ss << mvStrObjectLabels[label] << "(" << conf << ")" ;
			cv::Point2f left(data.at<float>(j, 2), data.at<float>(j, 3));
			cv::Point2f right(data.at<float>(j, 4), data.at<float>(j, 5));
			
			rectangle(dst,left, right, cv::Scalar(255, 255, 255));
			cv::putText(dst, ss.str(), cv::Point(left.x, left.y - 6), 1, 1.5, cv::Scalar::all(255));

			cv::Rect rect = cv::Rect(left, right);
			/*if (rect.area() > maxArea) {
				maxArea = rect.area();
				maxRect = rect;
			}*/

			////»ï°¢È­
			cv::Subdiv2D subdiv(rect);
			std::vector<cv::Point2f> vecPTs;
			for (int i = 0; i < vpMPs.size(); i++)
			{
				if (vpMPs[i] && rect.contains(F->mvKeys[i].pt)) {
					vecPTs.push_back(F->mvKeys[i].pt);
					subdiv.insert(F->mvKeys[i].pt);
				}
			}
			std::vector<cv::Vec6f> triangleList;
			subdiv.getTriangleList(triangleList);
			
			for (size_t i = 0,iend = triangleList.size(); i < iend; i++) 
			{
				cv::Vec6f t = triangleList[i];
				cv::Point2f pt1(t[0], t[1]);
				cv::Point2f pt2(t[2], t[3]);
				cv::Point2f pt3(t[4], t[5]);

				if (rect.contains(pt1) && rect.contains(pt2) && rect.contains(pt3)) {
					cv::line(objImg, pt1, pt2, cv::Scalar(255, 0, 0));
					cv::line(objImg, pt1, pt3, cv::Scalar(255, 0, 0));
					cv::line(objImg, pt3, pt2, cv::Scalar(255, 0, 0));
				}
			}
			////»ï°¢È­

			////Æ¯Â¡Á¡
			cv::Mat mask = cv::Mat::zeros(dst.size(), CV_8UC1);
			rectangle(mask, left, right, cv::Scalar(255, 255, 255),-1);
			cv::Mat objDesc;
			std::vector<cv::KeyPoint> vecObjKPs;
			detector->detectAndCompute(objImg, mask, vecObjKPs, objDesc);
			for (int i = 0, iend = vecObjKPs.size(); i < iend; i++) {
				cv::circle(objImg, vecObjKPs[i].pt, 2, cv::Scalar(255, 0, 0), 1);
			}
			////Æ¯Â¡Á¡
		}
		std::chrono::high_resolution_clock::time_point s2 = std::chrono::high_resolution_clock::now();
		auto d = std::chrono::duration_cast<std::chrono::milliseconds>(s2 - s1).count();
		float t2 = d / 1000.0;
		
		system->mpVisualizer->ResizeImage(dst, dst);
		system->mpVisualizer->SetOutputImage(dst, 2);

		//if (maxRect.area() > 400) {
		//	
		//	
		//	
		//	
		//	//rectangle(mask, cv::Point(maxRect.x, maxRect.y), cv::Point(maxRect.x + maxRect.width, maxRect.y + maxRect.height), cv::Scalar(255, 255, 255), -1);
		//	//rectangle(objImg, cv::Point(maxRect.x, maxRect.y), cv::Point(maxRect.x + maxRect.width, maxRect.y + maxRect.height), cv::Scalar(255, 255, 255));

		//	
		//	//Detector->detectAndCompute(objImg, cv::Mat(), vecObjKPs, objDesc);
		//	for (int i = 0, iend = vecObjKPs.size(); i < iend; i++) {
		//		cv::circle(objImg, vecObjKPs[i].pt, 2, cv::Scalar(255, 0, 0), 1);
		//	}

		//	
		//	
		//	
		//}
		std::cout << "Object Test = " << t2 << std::endl;
		cv::imshow("yolo", objImg); cv::waitKey(1);
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
			cv::normalize(depthImg, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
			cv::cvtColor(depth, depth, cv::COLOR_GRAY2BGR);
			/////save image
			/*std::stringstream sss;
			sss << "../../bin/img/" << pUser->userName << "/Depth/" << id << ".jpg";
			cv::imwrite(sss.str(), depth);*/
			/////save image
			system->mpVisualizer->ResizeImage(depth, depth);
			system->mpVisualizer->SetOutputImage(depth, 3);
		}
		
		return;

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

		////depth Á¤º¸ ÀúÀå ¹× Æ÷ÀÎÆ®¿Í ¿þÀÌÆ® Á¤º¸¸¦ Æ©ÇÃ·Î ÀúÀå
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
		////¿þÀÌÆ®¿Í Æ÷ÀÎÆ® Á¤º¸·Î Á¤·Ä
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
		////ÆÄ¶ó¸ÞÅÍ °Ë»ö ¹× µª½º Á¤º¸ º¹¿ø
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
			//////º¹¿ø È®ÀÎ
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
			//////º¹¿ø È®ÀÎ
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
	void Segmentator::RequestObjectDetection(std::string user, int id)
	{
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		std::stringstream ss;
		ss << "/Store?keyword=RequestObjectDetection&id=" << id << "&src=" << user;
		auto res = mpAPI->Send(ss.str(), "");
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
		mvStrLabels = Utils::Split(strLabel,",");
		mvStrObjectLabels = Utils::Split(strYoloObjectLabel, ",");

	}
}
