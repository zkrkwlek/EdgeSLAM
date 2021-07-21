#include <Utils.h>

namespace EdgeSLAM {

	cv::Mat Utils::SkewSymmetricMatrix(const cv::Mat &v)
	{
		return (cv::Mat_<float>(3, 3) << 0, -v.at<float>(2), v.at<float>(1),
			v.at<float>(2), 0, -v.at<float>(0),
			-v.at<float>(1), v.at<float>(0), 0);
	}

	cv::Mat Utils::ComputeF12(cv::Mat R1, cv::Mat t1, cv::Mat R2, cv::Mat t2, cv::Mat K1, cv::Mat K2)
	{
		cv::Mat R12 = R1*R2.t();
		cv::Mat t12 = -R1*R2.t()*t2 + t1;
		cv::Mat t12x = SkewSymmetricMatrix(t12);
		return K1.t().inv()*t12x*R12*K2.inv();
	}
}