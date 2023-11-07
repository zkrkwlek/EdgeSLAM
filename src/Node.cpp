#include <Node.h>

namespace EdgeSLAM {
	std::atomic<int> Node::nNodeID = 0;
	Node::Node():mnNodeId(++nNodeID),mAttr(NodeAttribute::None){}
	Node::~Node() {}
}
