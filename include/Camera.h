#ifndef EDGE_SLAM_CAMERA_H
#define EDGE_SLAM_CAMERA_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace EdgeSLAM {
	class Camera {
	public:
		Camera();
		Camera(int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, float _d5);
		virtual ~Camera();
	public:
		void Project();
		void Projects();
		void Unproject();
		void UnProjects();

		bool is_in_image(float x, float y, float z);
	public:
		int mnWidth, mnHeight;
		float fx, fy, cx, cy, invfx, invfy;
		cv::Mat K, D, Kinv;
		cv::Mat Kfluker;
		float u_min, u_max, v_min, v_max;
		bool bDistorted;
		cv::Mat Knew, map1, map2;
		static int mnGridSize;
		int mnGridRows, mnGridCols;
		float mfGridElementWidthInv, mfGridElementHeightInv;
		//std::vector<std::size_t>** mGrid;
	private:
		void undistort_image_bounds();
	};
}

#endif