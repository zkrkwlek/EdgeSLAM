#ifndef EDGE_SLAM_LOOP_TYPES_H
#define EDGE_SLAM_LOOP_TYPES_H
#pragma once

#include "g2o/types/types_seven_dof_expmap.h"

namespace EdgeSLAM {
	class KeyFrame;
	typedef std::pair<std::set<KeyFrame*>, int> ConsistentGroup;
	typedef std::map<KeyFrame*, g2o::Sim3, std::less<KeyFrame*>,
		Eigen::aligned_allocator<std::pair<const KeyFrame*, g2o::Sim3> > > KeyFrameAndPose;
}

#endif