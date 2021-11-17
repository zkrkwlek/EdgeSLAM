#ifndef EDGE_SLAM_OBJECT_FRAME_H
#define EDGE_SLAM_OBJECT_FRAME_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ConcurrentVector.h>

namespace EdgeSLAM {
	class MapPoint;
	class ObjectBox {
	public:
		ObjectBox();
		ObjectBox(int _label, float _conf, cv::Point2f pt1, cv::Point2f pt2);
		virtual ~ObjectBox();
	public:
		int label;
		float confidence;
		cv::Rect rect;
		ConcurrentVector<MapPoint*> vecMPs;
	private:
	};
	class ObjectFrame {
	public:
		ObjectFrame();
		virtual ~ObjectFrame();
	public:
		std::map<int, ObjectBox*> mapObjects;
	private:
		
	};
}
#endif