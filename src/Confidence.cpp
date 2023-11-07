#include <Confidence.h>
#include <MapPoint.h>

Confidence::Confidence():d1(10000.0),d2(10000.0),d3(10000.0),w1(0.3),w2(0.3),w3(0.4){}
Confidence::Confidence(float _w1, float _w2, float _w3): d1(0.0), d2(0.0), d3(0.0), w1(_w1),w2(_w2),w3(_w3){}
Confidence::~Confidence() {}

float Confidence::CalcConfidence(){}

float Confidence::CalcConfidence(cv::Point2f pt1, cv::Point2f pt2) {
	auto diffPt1 = pt1 - pt2;
	d1 = sqrt(diffPt1.dot(diffPt1));
	return d1;
}

float Confidence::CalcConfidence(float _dist) {
	d3 = _dist;
	return d3;
}