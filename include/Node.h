#ifndef DYNAMIC_SLAM_NODE_H
#define DYNAMIC_SLAM_NODE_H
#pragma once

#include <ConcurrentSet.h>

enum class NodeAttribute {
	None = 0,
	ON = 1
};

namespace EdgeSLAM {

	class Node {
	public:
		Node();
		virtual~Node();

	public:
		static std::atomic<int> nNodeID;
		int mnNodeId;
		NodeAttribute mAttr;
		ConcurrentSet<Node*> mSetConnectedNodes;
	private:
		
	};
}

#endif