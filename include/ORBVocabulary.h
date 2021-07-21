#ifndef EDGE_SLAM_ORBVOCABULARY_H
#define EDGE_SLAM_ORBVOCABULARY_H
#pragma once
//#include "DBoW3.h"

#include <DBoW2/DBoW2/FORB.h>
#include <DBoW2/DBoW2/TemplatedVocabulary.h>

namespace EdgeSLAM
{
	typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;
	//typedef DBoW3::Vocabulary ORBVocabulary;

}
#endif