#ifndef EDGE_SLAM_SEGMENTATOR_H
#define EDGE_SLAM_SEGMENTATOR_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ThreadPool.h>
#include <WebAPI.h>
namespace EdgeSLAM {
	
	enum class ObjectLabel {
		WALL = 1,
		FLOOR = 4,
		CEIL = 6
	};

	class SLAM;
	class Frame;
	class User;
	class Segmentator {
	public:
		Segmentator();
		virtual ~Segmentator();
		static void ProcessSegmentation(ThreadPool::ThreadPool* pool, SLAM* system, std::string user,int id);
		static void ProcessDepthEstimation(ThreadPool::ThreadPool* pool, SLAM* system, std::string user, int id);
		static void RequestSegmentation(std::string user,int id);
		static void Init();
	public:
		static std::vector<cv::Vec3b> mvObjectLabelColors;
	private:

	};
}

#endif