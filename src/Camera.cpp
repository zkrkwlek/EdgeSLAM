#include <Camera.h>

namespace EdgeSLAM {
	int Camera::mnGridSize = 10;
	Camera::Camera(){}
	Camera::~Camera(){}
	Camera::Camera(int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4)
		:mnWidth(_w), mnHeight(_h), fx(_fx), fy(_fy), cx(_cx), cy(_cy), invfx(1.0/fx), invfy(1.0/fy)
	{
		K = cv::Mat::eye(3, 3, CV_32FC1);
		K.at<float>(0, 0) = fx;
		K.at<float>(1, 1) = fy;
		K.at<float>(0, 2) = cx;
		K.at<float>(1, 2) = cy;

		Kinv = cv::Mat::eye(3, 3, CV_32FC1);
		Kinv.at<float>(0, 0) = 1.0 / fx;
		Kinv.at<float>(1, 1) = 1.0 / fy;
		Kinv.at<float>(0, 2) = -cx / fx;
		Kinv.at<float>(1, 2) = -cy / fy;

		D = cv::Mat::zeros(4, 1, CV_32FC1);
		D.at<float>(0) = _d1;
		D.at<float>(1) = _d2;
		D.at<float>(2) = _d3;
		D.at<float>(3) = _d4;

		bDistorted = sqrt(D.dot(D)) > 1e-10;
		u_min = 0.0;
		u_max = (float)mnWidth;
		v_min = 0.0;
		v_max = (float)mnHeight;
		if (bDistorted) {
			undistort_image_bounds();
		}
		mnGridCols = mnWidth / mnGridSize;
		mnGridRows = mnHeight / mnGridSize;

		mfGridElementWidthInv  = static_cast<float>(mnGridCols) / (u_max - u_min);
		mfGridElementHeightInv = static_cast<float>(mnGridRows) / (v_max - v_min);

	}

	bool Camera::is_in_image(float x, float y, float z) {
		return x > u_min && x < u_max && y > v_min && y < v_max && z > 0.0;
	}

	void Camera::undistort_image_bounds() {
		cv::Mat mat(4, 2, CV_32F);
		mat.at<float>(0, 0) = u_min;		mat.at<float>(0, 1) = v_min;
		mat.at<float>(1, 0) = u_max;		mat.at<float>(1, 1) = v_min;
		mat.at<float>(2, 0) = u_min;		mat.at<float>(2, 1) = v_max;
		mat.at<float>(3, 0) = u_max;		mat.at<float>(3, 1) = v_max;

		// Undistort corners
		mat = mat.reshape(2);
		cv::undistortPoints(mat, mat, K, D, cv::Mat(), K);
		mat = mat.reshape(1);

		u_min = std::min(mat.at<float>(0, 0), mat.at<float>(2, 0));
		u_max = std::max(mat.at<float>(1, 0), mat.at<float>(3, 0));
		v_min = std::min(mat.at<float>(0, 1), mat.at<float>(1, 1));
		v_max = std::max(mat.at<float>(2, 1), mat.at<float>(3, 1));
	}
}