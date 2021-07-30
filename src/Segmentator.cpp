#include <Segmentator.h>
#include <SLAM.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <User.h>
#include <Camera.h>
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
		ss << "/Load?keyword=Segmentation"  << "&id=" << id;
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
		auto KF = pUser->mapKeyFrames[F->mnKeyFrameId];

		cv::Mat resized_test;
		cv::resize(segcolor, resized_test, cv::Size(segcolor.cols / 2, segcolor.rows / 2));
		system->mpVisualizer->SetOutputImage(resized_test, 1);

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
		ss << "/Load?keyword=DepthEstimation" << "&id=" << id;
		WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		auto res = mpAPI->Send(ss.str(), "");
		int n2 = res.size();

		int w = pUser->mpCamera->mnWidth;
		int h = pUser->mpCamera->mnHeight;
		cv::Mat labeled = cv::Mat::zeros(h, w, CV_32FC1);
		std::memcpy(labeled.data, res.data(), res.size());

		////segmentation image
		cv::Mat dpehtImg = cv::Mat::zeros(h, w, CV_8UC3);
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

		auto F = pUser->mapFrames[id];
		auto KF = pUser->mapKeyFrames[F->mnKeyFrameId];
		
		cv::Mat depth, resized_test;
		cv::normalize(labeled, depth, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		cv::cvtColor(depth, depth, cv::COLOR_GRAY2BGR);
		cv::resize(depth, resized_test, cv::Size(depth.cols / 2, depth.rows / 2));
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
