#include <ObjectFrame.h>

namespace EdgeSLAM {

	ObjectBox::ObjectBox() {}
	ObjectBox::ObjectBox(int _label, float _conf, cv::Point2f pt1, cv::Point2f pt2):label(_label), confidence(_conf), rect(cv::Rect(pt1, pt2)){}
	ObjectBox::~ObjectBox(){}

	ObjectFrame::ObjectFrame() {}
	ObjectFrame::~ObjectFrame() {}
}