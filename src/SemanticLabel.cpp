#include <SemanticLabel.h>

namespace EdgeSLAM {
	ObjectLabel::ObjectLabel(){}
	ObjectLabel::~ObjectLabel() {
		LabelCount.Release();
	}
	SemanticLabel::SemanticLabel() {}
	SemanticLabel::~SemanticLabel() {
		LabelCount.Release();
	}
}
