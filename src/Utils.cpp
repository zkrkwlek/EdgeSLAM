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
	cv::Mat Utils::RotationMatrixFromEulerAngle(float a, char c) {
		cv::Mat res = cv::Mat::eye(3, 3, CV_32FC1);

		//input¿∫ radian
		float cx = cos(a);
		float sx = sin(a);

		if (c == 'x' || c == 'X') {
			res.at<float>(1, 1) = cx;
			res.at<float>(1, 2) = -sx;
			res.at<float>(2, 1) = sx;
			res.at<float>(2, 2) = cx;
		}
		else if (c == 'y' || c == 'Y') {
			res.at<float>(0, 0) = cx;
			res.at<float>(0, 2) = sx;
			res.at<float>(2, 0) = -sx;
			res.at<float>(2, 2) = cx;
		}
		else if (c == 'z' || c == 'Z') {
			res.at<float>(0, 0) = cx;
			res.at<float>(0, 1) = -sx;
			res.at<float>(1, 0) = sx;
			res.at<float>(1, 1) = cx;
		}
		return res;
	}
	cv::Mat Utils::RotationMatrixFromEulerAngles(float a, float b, float c, std::string str) {
		auto cstr = str.c_str();
		cv::Mat R1 = RotationMatrixFromEulerAngle(a, cstr[0]);
		cv::Mat R2 = RotationMatrixFromEulerAngle(b, cstr[1]);
		cv::Mat R3 = RotationMatrixFromEulerAngle(c, cstr[2]);
		return R1*R2*R3;
	}
}