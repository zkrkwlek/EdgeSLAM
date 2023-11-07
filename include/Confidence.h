#ifndef DYNAMIC_SLAM_CONFIDENCE_H
#define DYNAMIC_SLAM_CONFIDENCE_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace EdgeSLAM {
	class MapPoint;
}

class Confidence {
public:
	Confidence();
	Confidence(float _w1, float _w2, float _w3);
	virtual ~Confidence();

public:
	float CalcConfidence();
	float CalcConfidence(cv::Point2f pt1, cv::Point2f pt2); //2d image
	float CalcConfidence(float _dist); //plane
private:
	//float CalcConfidence1();
	//float CalcConfidence2();
	//float CalcConfidence3();
public:
	float d1, d2, d3;
	float w1, w2, w3;
};

#endif