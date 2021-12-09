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
#include <Segmentator.h>
#include <SearchPoints.h>
#include <Converter.h>

namespace EdgeSLAM {
	SLAM::SLAM():pool(){
	
	}
	SLAM::SLAM(ThreadPool::ThreadPool* _pool):pool(_pool){
		Init();
	}
	SLAM::~SLAM(){}
	FeatureDetector* Frame::Detector;
	FeatureDetector* Segmentator::Detector;
	FeatureTracker* Segmentator::Matcher;
	FeatureTracker* SearchPoints::Matcher;
	
	FeatureTracker* MapPoint::mpDist;
	DBoW3::Vocabulary* KeyFrame::mpVoc;
	DBoW3::Vocabulary* Frame::mpVoc;
	/*ORBVocabulary* KeyFrame::mpVoc;
	ORBVocabulary* Frame::mpVoc;*/
	void SLAM::Init() {
		LoadVocabulary();
		
		////이거 수정 필요
		//LoadProcessingTime();

		Segmentator::Init();
		
		mpInitializer = new Initializer();
		mpTracker = new Tracker();
		mpFeatureTracker = new FlannFeatureTracker(1500);
		mpLocalMapper = new LocalMapper();
		mpLoopCloser = new LoopCloser();
		mpVisualizer = new Visualizer(this);

		//set method
		KeyFrame::mpVoc = mpDBoWVoc;
		Frame::mpVoc = mpDBoWVoc;
		Frame::Detector = mpFeatureTracker->detector;
		Segmentator::Detector = mpFeatureTracker->detector;
		Segmentator::Matcher = mpFeatureTracker;
		MapPoint::mpDist = mpFeatureTracker;
		SearchPoints::Matcher = mpFeatureTracker;
		mpInitializer->mpFeatureTracker = mpFeatureTracker;

		mnVisID = 0;
	}
	void SLAM::LoadVocabulary() {
		mpDBoWVoc = new DBoW3::Vocabulary();
		mpDBoWVoc->load("../bin/data/orbvoc.dbow3");
		/*mpDBoWVoc = new ORBVocabulary();
		mpDBoWVoc->loadFromBinaryFile("../../bin/data/ORBvoc.bin");*/
	}
	void SLAM::Track(cv::Mat im, int id, User* user, double ts) {
		
	}

	void SLAM::Track(int id, User* user, double ts) {
		pool->EnqueueJob(Tracker::Track, pool, this, id, user, ts);
	}
	bool bVis = false;
	void SLAM::InitVisualizer(std::string user, std::string name, int w, int h) {
		/*if (bVis)
			return;*/
		if(!bVis){
			mpVisualizer->Init(w, h);
			mptVisualizer = new std::thread(&EdgeSLAM::Visualizer::Run, mpVisualizer);
			mpVisualizer->SetMap(GetMap(name));
			mpVisualizer->strMapName = name;
			bVis = true;
		}
		//mpVisualizer->AddUser(GetUser(user));
	}
	void SLAM::ProcessContentGeneration(std::string user, int id) {
		if (!CheckUser(user))
			return;
		auto pUser = GetUser(user);
		pool->EnqueueJob(Segmentator::ProcessContentGeneration, this, pUser, id);
	}
	void SLAM::ProcessSegmentation(std::string user,int id) {
		pool->EnqueueJob(Segmentator::ProcessSegmentation, pool, this, user, id);
	}
	void SLAM::ProcessObjectDetection(std::string user, int id) {
		pool->EnqueueJob(Segmentator::ProcessObjectDetection, pool, this, user, id);
	}
	void SLAM::ProcessDepthEstimation(std::string user, int id) {
		pool->EnqueueJob(Segmentator::ProcessDepthEstimation, pool, this, user, id);
	}
	//////////////////Multi User and Map
	void SLAM::CreateMap(std::string name) {
		auto pNewMap = new Map(mpDBoWVoc);
		AddMap(name, pNewMap);
	}
	void SLAM::CreateUser(std::string _user, std::string _map, int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, float _d5, int quality, bool _b, bool _bTracking, bool _bimu){
		auto pNewUser = new User(_user, _map, _w, _h, _fx, _fy, _cx, _cy, _d1, _d2, _d3, _d4, _d5, quality, _b, _bTracking, _bimu);
		pNewUser->mpMap = GetMap(_map);
		AddUser(_user, pNewUser);
	}

	bool SLAM::CheckMap(std::string str){
		return Maps.Count(str) > 0;
		/*std::unique_lock<std::mutex> lock(mMutexMapList);
		if (mmpMapList.count(str))
			return true;
		return false;*/
	}
	bool SLAM::CheckUser(std::string str){
		return Users.Count(str) > 0;
		/*std::unique_lock<std::mutex> lock(mMutexUserList);
		if (mmpConnectedUserList.count(str)) {
			return true;
		}
		return false;*/
	}
	void SLAM::UpdateDeviceGyroSensor(std::string user, int id) {
		if (!CheckUser(user))
			return;
		auto pUser = GetUser(user);
		pool->EnqueueJob(Tracker::UpdateDeviceGyro, this, pUser, id);
		
	}
	void SLAM::UpdateDevicePosition(std::string user, int id) {
		if (!CheckUser(user))
			return;
		auto pUser = GetUser(user);
		pool->EnqueueJob(Segmentator::ProcessDevicePosition, this, pUser, id);
		//////
		//std::stringstream ss;
		//ss << "/Load?keyword=DevicePosition" << "&id=" << id << "&src=" << user;
		//WebAPI* mpAPI = new WebAPI("143.248.6.143", 35005);
		//auto res = mpAPI->Send(ss.str(), "");
		//int n2 = res.size();
		//
		//cv::Mat pos = cv::Mat::zeros(12, 1, CV_32FC1);
		//std::memcpy(pos.data, res.data(), res.size());
		//std::cout << pos.t() << std::endl;

		//pUser->AddDevicePosition(pos);
	}
	void SLAM::AddUser(std::string id, User* user) {
		SetUserVisID(user);
		Users.Update(id, user);
		/*std::unique_lock<std::mutex> lock(mMutexUserList);
		SetUserVisID(user);
		mmpConnectedUserList[id] = user;*/
	}
	User* SLAM::GetUser(std::string id) {
		if(Users.Count(id))
			return Users.Get(id);
		return nullptr;
		/*std::unique_lock<std::mutex> lock(mMutexUserList);
		if (mmpConnectedUserList.count(id)) {
			return mmpConnectedUserList[id];
		}
		return nullptr;*/
	}
	std::vector<User*> SLAM::GetAllUsersInMap(std::string map) {
		std::vector<User*> res;
		if (!CheckMap(map))
			return res;
		////string compare

		std::map<std::string, User*> mapUserLists = Users.Get();
		/*{
			std::unique_lock<std::mutex> lock(mMutexUserList); 
			mapUserLists = mmpConnectedUserList;
		}*/
		for (auto iter = mapUserLists.begin(), iend = mapUserLists.end(); iter != iend; iter++) {
			auto user = iter->second;
			if (user->mapName != map)
				continue;
			res.push_back(user);
		}
		return res;
	}
	void SLAM::RemoveUser(std::string id) {
		bool bDelete = false;
		{
			if (Users.Count(id)) {
				auto user = Users.Get(id);
				Users.Erase(id);
				delete user;
				bDelete = true;
			}
			/*std::unique_lock<std::mutex> lock(mMutexUserList);
			if (mmpConnectedUserList.count(id)) {
				auto user = mmpConnectedUserList[id];
				mmpConnectedUserList.erase(id);
				bDelete = true;
			}*/
		}
		if(bDelete)
			UpdateUserVisID();
	}
	int SLAM::GetConnectedDevice() {
		return Users.Size();
	}
	void SLAM::SetUserVisID(User* user){
		std::unique_lock<std::mutex> lock(mMutexVisID);
		if (user->mbMapping) {
			return;
		}
		user->SetVisID(mnVisID);
		mnVisID++;
		std::cout << user->userName << "=" << user->GetVisID() << std::endl;
	}
	void SLAM::UpdateUserVisID(){
		std::unique_lock<std::mutex> lock(mMutexVisID);
		std::map<std::string, User*> mapUserLists = Users.Get();
		/*{
			std::unique_lock<std::mutex> lock(mMutexUserList);
			mapUserLists = mmpConnectedUserList;
		}*/
		mnVisID = 0;
		for (auto iter = mapUserLists.begin(), iend = mapUserLists.end(); iter != iend; iter++) {
			auto user = iter->second;
			if (user->mbMapping)
				continue;
			user->SetVisID(mnVisID);
			mnVisID++;
		}
	}

	void SLAM::VisualizeImage(cv::Mat src, int vid) {
		mpVisualizer->ResizeImage(src, src);
		mpVisualizer->SetOutputImage(src, vid);
	}

	/*
	void SLAM::UpdateTrackingTime(float ts){
		std::unique_lock<std::mutex> lock(mMutexTrackingTime);
		nTotalTrack++;
		fSumTrack += ts;
		fSumTrack2 += (ts*ts);
	}
	void SLAM::UpdateRelocTime(float ts){
		std::unique_lock<std::mutex> lock(mMutexRelocTime);
		nTotalReloc++;
		fSumReloc += ts;
		fSumReloc2 += (ts*ts);
	}
	void SLAM::UpdateMappingTime(float ts){
		std::unique_lock<std::mutex> lock(mMutexMappingTime);
		nTotalMapping++;
		fSumMapping += ts;
		fSumMapping2 += (ts*ts);
	}
	*/
	void SLAM::InitProcessingTime() {
		//ProcessingTime
		for (int i = 1; i < 9; i++) {
			std::map<std::string, ProcessTime*> vec;
			std::stringstream ss;
			ss << "../bin/time/tracking_" << i << ".txt";
			vec.insert(std::make_pair("tracking", new ProcessTime(ss.str())));
			ss.str("");
			ss << "../bin/time/mapping_" << i << ".txt";
			vec.insert(std::make_pair("mapping", new ProcessTime(ss.str())));
			ss.str("");
			ss << "../bin/time/reloc_" << i << ".txt";
			vec.insert(std::make_pair("reloc", new ProcessTime(ss.str())));
			ss.str("");
			ss << "../bin/time/download_" << i << ".txt";
			vec.insert(std::make_pair("download", new ProcessTime(ss.str())));
			ss.str("");
			ss << "../bin/time/upload" << i << ".txt";
			vec.insert(std::make_pair("upload", new ProcessTime(ss.str())));
			ss.str("");
			ProcessingTime.Update(i, vec);
		}
	}
	void SLAM::SaveProcessingTime() {
		
		auto AllData = ProcessingTime.Get();
		for (auto iter = AllData.begin(), iend = AllData.end(); iter != iend; iter++) {
			auto vec = iter->second;
			for (auto jter = vec.begin(), jend = vec.end(); jter != jend; jter++) {
				auto temp = jter->second;
				temp->update();
				temp->save(temp->getname());
			}
		}
	}
	void SLAM::LoadProcessingTime(){
		
		for (int i = 1; i < 9; i++) {
			std::map<std::string,ProcessTime*> vec;
			std::stringstream ss;
			ss << "../bin/time/tracking_" << i << ".txt";
			auto p1 = new ProcessTime(ss.str());
			p1->load(ss.str());
			ss.str("");
			ss << "../bin/time/mapping_" << i << ".txt";
			auto p2 = new ProcessTime(ss.str());
			p2->load(ss.str());
			ss.str("");
			ss << "../bin/time/reloc_" << i << ".txt";
			auto p3 = new ProcessTime(ss.str());
			p3->load(ss.str());
			ss.str("");
			ss << "../bin/time/download_" << i << ".txt";
			auto p4 = new ProcessTime(ss.str());
			p4->load(ss.str());
			ss.str("");
			ss << "../bin/time/upload_" << i << ".txt";
			auto p5 = new ProcessTime(ss.str());
			p5->load(ss.str());
			vec.insert(std::make_pair("tracking", p1));
			vec.insert(std::make_pair("mapping", p2));
			vec.insert(std::make_pair("reloc", p3));
			vec.insert(std::make_pair("download", p4));
			vec.insert(std::make_pair("upload", p5));
			ProcessingTime.Update(i, vec);
		}

	}
	void SLAM::SaveTrajectory(User* user) {
		__time64_t long_time;
		_time64(&long_time);
		struct tm newtime;
		_localtime64_s(&newtime, &long_time);

		auto pMap = GetMap(user->mapName);
		auto vpKFs =  pMap->GetAllKeyFrames();

		std::stringstream ss;
		ss << "../bin/trajectory/" << user->mapName <<"_"<<user->userName<<"_"<<user->mnQuality<<"_"<< newtime.tm_year + 1900 <<"_"<< newtime.tm_mon+1<<"_"<< newtime.tm_mday<<"_"<< newtime.tm_hour<<"_"<< newtime.tm_min<< ".txt";
		std::ofstream f;
		f.open(ss.str().c_str());
		f << std::fixed;

		if (user->mbMapping) {
			for (int i = 0; i < vpKFs.size(); i++) {
				auto pKF = vpKFs[i];

				cv::Mat R = pKF->GetRotation();
				cv::Mat t = pKF->GetTranslation();
				R = R.t(); //inverse
				t = -R*t;  //camera center
				std::vector<float> q = Converter::toQuaternion(R);
				f << std::setprecision(6) << pKF->mdTimeStamp << std::setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
					<< " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
			}
			
		}
		else{
			
			for (int i = 0; i < user->vecTrajectories.size(); i++) {
				cv::Mat R = user->vecTrajectories[i].rowRange(0, 3).colRange(0, 3);
				cv::Mat t = user->vecTrajectories[i].rowRange(0, 3).col(3);
				R = R.t(); //inverse
				t = -R*t;  //camera center
				std::vector<float> q = Converter::toQuaternion(R);
				f << std::setprecision(6) << user->vecTimestamps[i] << std::setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
					<< " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
			}
		}
		f.close();
		
	}
	void SLAM::AddMap(std::string name, Map* pMap) {
		Maps.Update(name, pMap);
		/*std::unique_lock<std::mutex> lock(mMutexMapList);
		mmpMapList[name] = pMap;*/
	}
	Map* SLAM::GetMap(std::string name) {
		if (Maps.Count(name))
			return Maps.Get(name);
		return nullptr;
		/*std::unique_lock<std::mutex> lock(mMutexMapList);
		if (mmpMapList.count(name)) {
			return mmpMapList[name];
		}
		return nullptr;*/
	}
	void SLAM::RemoveMap(std::string name) {
		if (Maps.Count(name))
			Maps.Erase(name);
		/*std::unique_lock<std::mutex> lock(mMutexMapList);
		mmpMapList.erase(name);*/
	}
}