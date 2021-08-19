#include <Segmentator.h>
#include <SLAM.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <User.h>
#include <Camera.h>
#include <Map.h>
#include <Visualizer.h>

namespace EdgeSLAM {

	std::vector<cv::Vec3b>Segmentator:: mvObjectLabelColors;
	Segmentator::Segmentator() {

	}
	Segmentator::~Segmentator() {

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
				}
			}
		}

		auto F = pUser->mapFrames[id];
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
			int nData = nTotal;
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
			//복원 확인
			cv::Mat invK = F->InvK.clone();
			cv::Mat Rinv, Tinv;
			cv::Mat Pinv = F->GetPoseInverse();
			Rinv = Pinv.colRange(0, 3).rowRange(0, 3);
			Tinv = Pinv.col(3).rowRange(0, 3);
			map->ClearDepthMPs();
			int inc = 5;
			for (size_t row = inc, rows = depthImg.rows; row < rows; row+= inc) {
				for (size_t col = inc, cols = depthImg.cols; col < cols; col+= inc) {
					cv::Point2f pt(col, row);
					float depth = depthImg.at<float>(pt);
					if (depth < 0.0001)
						continue;
					cv::Mat a = Rinv*(invK*(cv::Mat_<float>(3, 1) << pt.x, pt.y, 1.0)*depth) + Tinv;
					map->AddDepthMP(a);
				}
			}
		}
		//////////////////
		
		//auto KF = pUser->mapKeyFrames[F->mnKeyFrameId];
		
		cv::Mat depth, resized_test;
		cv::normalize(depthImg, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::cvtColor(depth, depth, cv::COLOR_GRAY2BGR);
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
		std::cout << "request segmentation" << std::endl;
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
	}
}
