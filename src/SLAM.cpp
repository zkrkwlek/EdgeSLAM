#include <SLAM.h>
#include <Map.h>
#include <User.h>
#include <Initializer.h>
#include <Tracker.h>
#include <LocalMapper.h>
#include <LoopCloser.h>
#include <FeatureTracker.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <Visualizer.h>


namespace EdgeSLAM {

	SLAM::SLAM(){
		Init();
	}
	SLAM::~SLAM(){}

	FeatureDetector* Frame::detector;
	FeatureTracker* Frame::matcher;
	FeatureTracker* KeyFrame::matcher;
	FeatureTracker* MapPoint::mpDist;
	DBoW3::Vocabulary* KeyFrame::mpVoc;
	DBoW3::Vocabulary* Frame::mpVoc;
	/*ORBVocabulary* KeyFrame::mpVoc;
	ORBVocabulary* Frame::mpVoc;*/
	void SLAM::Init() {
		LoadVocabulary();
		pool = new ThreadPool::ThreadPool(16);
		mpInitializer = new Initializer();
		mpTracker = new Tracker();
		mpFeatureTracker = new FlannFeatureTracker(1000);
		mpLocalMapper = new LocalMapper();
		mpLoopCloser = new LoopCloser(mpDBoWVoc, true);
		mpVisualizer = new Visualizer(this);

		//set method
		KeyFrame::mpVoc = mpDBoWVoc;
		Frame::mpVoc = mpDBoWVoc;
		Frame::detector = mpFeatureTracker->detector;
		Frame::matcher = mpFeatureTracker;
		KeyFrame::matcher = mpFeatureTracker;
		MapPoint::mpDist = mpFeatureTracker;
		mpInitializer->mpFeatureTracker = mpFeatureTracker;
	}
	void SLAM::LoadVocabulary() {
		mpDBoWVoc = new DBoW3::Vocabulary();
		mpDBoWVoc->load("../../bin/data/orbvoc.dbow3");
		/*mpDBoWVoc = new ORBVocabulary();
		mpDBoWVoc->loadFromBinaryFile("../../bin/data/ORBvoc.bin");*/
	}
	void SLAM::Track(cv::Mat im, int id, User* user, double ts) {
		pool->EnqueueJob(Tracker::Track, pool, this, im, id, user, ts);
	}
	bool bVis = false;
	void SLAM::InitVisualizer(std::string user, std::string name, int w, int h) {
		if (bVis)
			return;
		bVis = true;
		mpVisualizer->Init(w, h);
		mptVisualizer = new std::thread(&EdgeSLAM::Visualizer::Run, mpVisualizer);
		mpVisualizer->SetMap(GetMap(name));
		mpVisualizer->AddUser(GetUser(user));
	}

	//////////////////Multi User and Map
	void SLAM::CreateMap(std::string name) {
		auto pNewMap = new Map(mpDBoWVoc);
		AddMap(name, pNewMap);
	}
	void SLAM::CreateUser(std::string _user, std::string _map, int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, bool _b){
		auto pNewUser = new User(_user, _map, _w, _h, _fx, _fy, _cx, _cy, _d1, _d2, _d3, _d4, _b);
		pNewUser->mpMap = GetMap(_map);
		AddUser(_user, pNewUser);
	}

	bool SLAM::CheckMap(std::string str){
		std::unique_lock<std::mutex> lock(mMutexMapList);
		if (mmpMapList.count(str))
			return true;
		return false;
	}
	bool SLAM::CheckUser(std::string str){
		std::unique_lock<std::mutex> lock(mMutexUserList);
		if (mmpConnectedUserList.count(str)) {
			return true;
		}
		return false;
	}
	void SLAM::AddUser(std::string id, User* user) {
		std::unique_lock<std::mutex> lock(mMutexUserList);
		/*for (int i = 0; i < mapVisID.size(); i++) {
			if (!mapVisID[i].second) {
				mapVisID[i].second = true;
				user->mnVisID = mapVisID[i].first;
				break;
			}
		}*/
		mmpConnectedUserList[id] = user;
	}
	User* SLAM::GetUser(std::string id) {
		std::unique_lock<std::mutex> lock(mMutexUserList);
		if (mmpConnectedUserList.count(id)) {
			return mmpConnectedUserList[id];
		}
		return nullptr;
	}
	void SLAM::RemoveUser(std::string id) {
		std::unique_lock<std::mutex> lock(mMutexUserList);
		if (mmpConnectedUserList.count(id)) {
			auto user = mmpConnectedUserList[id];
			/*if (user->mnVisID >= 0) {
				mapVisID[user->mnVisID].second = false;
			}*/
			mmpConnectedUserList.erase(id);

		}
	}
	void SLAM::AddMap(std::string name, Map* pMap) {
		std::unique_lock<std::mutex> lock(mMutexMapList);
		mmpMapList[name] = pMap;
	}
	Map* SLAM::GetMap(std::string name) {
		std::unique_lock<std::mutex> lock(mMutexMapList);
		if (mmpMapList.count(name)) {
			return mmpMapList[name];
		}
		return nullptr;
	}
	void SLAM::RemoveMap(std::string name) {
		std::unique_lock<std::mutex> lock(mMutexMapList);
		mmpMapList.erase(name);
	}
}