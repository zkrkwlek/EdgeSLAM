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
#include <io.h>
#include <direct.h>

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
		//mpVisualizer = new Visualizer(this);

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
	
	void SLAM::TrackOXR(int id, std::string user, double ts) {
		pool->EnqueueJob(Tracker::TrackWithKnownPose, pool, this, id, user, ts);
	}

	void SLAM::Track(int id, std::string user, double ts) {
		pool->EnqueueJob(Tracker::Track, pool, this, id, user, ts);
	}
	
	void SLAM::InitVisualizer(std::string user, std::string name, int w, int h) {
		/*if (bVis)
			return;*/
		auto pMap = GetMap(name);
		if (!pMap->mbVisualized) {
			pMap->mpVisualizer = new Visualizer(this);
			pMap->mpVisualizer->Init(w, h);
			//mptVisualizer = new std::thread(&EdgeSLAM::Visualizer::Run, mpVisualizer);
			new std::thread(&EdgeSLAM::Visualizer::Run, pMap->mpVisualizer);
			pMap->mpVisualizer->SetMap(GetMap(name));
			pMap->mpVisualizer->strMapName = name;
			pMap->mbVisualized = true;
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
	void SLAM::CreateMap(std::string name, int nq) {
		auto pNewMap = new Map(mpDBoWVoc);
		AddMap(name, pNewMap);
		MapQuality.Update(name, nq);
	}
	void SLAM::CreateUser(std::string _user, std::string _map, int _w, int _h, float _fx, float _fy, float _cx, float _cy, float _d1, float _d2, float _d3, float _d4, float _d5, int quality, int nskip, bool _b, bool _bTracking, bool _bBaseLocalMap, bool _bimu, bool _bGBA, bool _bReset, bool _bsave, bool _basync){
		auto pNewUser = new User(_user, _map, _w, _h, _fx, _fy, _cx, _cy, _d1, _d2, _d3, _d4, _d5, quality, nskip, _b, _bTracking, _bBaseLocalMap, _bimu, _bGBA, _bReset, _bsave, _basync);
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
	int  SLAM::CountUser() {
		return Users.Size();
	}
	
	void SLAM::UpdateDeviceGyroSensor(std::string user, int id) {
		if (!CheckUser(user))
			return;
		pool->EnqueueJob(Tracker::UpdateDeviceGyro, this, user, id);
	}
	void SLAM::UpdateDevicePosition(std::string user, int id, double ts) {
		if (!CheckUser(user))
			return;
		pool->EnqueueJob(Tracker::ProcessDevicePosition, this, user, id, ts);
	}
	void SLAM::AddUser(std::string id, User* user) {
		SetUserVisID(user);
		Users.Update(id, user);
		/*std::unique_lock<std::mutex> lock(mMutexUserList);
		SetUserVisID(user);
		mmpConnectedUserList[id] = user;*/
	}
	User* SLAM::GetUser(std::string id) {
		if(Users.Count(id)){
			auto pUser = Users.Get(id);
			if(!pUser->mbRemoved)
				return pUser;
		}
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
			if (user->mapName != map || user->mbRemoved)
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
				if (user->mbRemoved) {
					std::cout << "Doing removing process = " <<id<< std::endl;
					return;
				}
				user->mbRemoved = true;
				
				/*if (user->mnUsed > 0) {
					std::cout << "Tracking status = " << user->mnDebugTrack << std::endl;
					std::cout << "Sematic status = " << user->mnDebugSeg << std::endl;
					std::cout << "AR status = " << user->mnDebugAR << std::endl;
					std::cout << "Label status = " << user->mnDebugLabel << std::endl;
					std::cout << "Plane status = " << user->mnDebugPlane << std::endl;
				}*/
				int count = 0;
				while (user->mnUsed > 0){
					count++;
					if (count %= 200) {
						std::cout << "Tracking status = " << user->mnDebugTrack << std::endl;
						std::cout << "Sematic status = " << user->mnDebugSeg << std::endl;
						std::cout << "AR status = " << user->mnDebugAR << std::endl;
						std::cout << "Label status = " << user->mnDebugLabel << std::endl;
						std::cout << "Plane status = " << user->mnDebugPlane << std::endl;
					}
					continue;
				}
				Users.Erase(id);
				delete user;
				bDelete = true;
				std::cout << "Remove user = " << id << std::endl;
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
		/*if (user->mbMapping) {
			return;
		}*/
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

	//t일 때 트래킹 출력
	//f일 때 나머지 출력
	//왼쪽은 시각화, 오른쪽은 기기 위치 출력하기
	void SLAM::VisualizeImage(std::string mapName, cv::Mat src, int vid) {
		auto pMap = GetMap(mapName);
		pMap->mpVisualizer->ResizeImage(src, src);
		pMap->mpVisualizer->SetOutputImage(src, vid);
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

		std::map<int, ProcessTime*> vec1;
		std::map<int, ProcessTime*> vec2;
		std::map<int, ProcessTime*> vec3;
		std::map<int, ProcessTime*> vec4;
		std::map<int, ProcessTime*> vec5;

		for (int i = 1; i < 9; i++) {
			vec1.insert(std::make_pair(i, new ProcessTime("tracking", i)));
			vec2.insert(std::make_pair(i, new ProcessTime("mapping",i)));
			vec3.insert(std::make_pair(i, new ProcessTime("reloc", i)));
			vec4.insert(std::make_pair(i, new ProcessTime("download",i)));
			vec5.insert(std::make_pair(i, new ProcessTime("upload", i)));
		}

		ProcessingTime.Update("tracking", vec1);
		ProcessingTime.Update("mapping", vec2);
		ProcessingTime.Update("reloc", vec3);
		ProcessingTime.Update("download", vec4);
		ProcessingTime.Update("upload", vec5);

		/*for (int i = 1; i < 9; i++) {
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
		}*/

		std::map<int, Ratio*> vec;
		for (int i = 1; i < 15; i++) {
			std::stringstream ss;
			ss << "../bin/time/skipframe_" << i << ".txt";
			//vec.insert(std::make_pair(i, new Ratio(ss.str())));
		}
		SuccessRatio.Update("skipframe", vec);
	}
	void SLAM::SaveProcessingTime() {
		
		{
			std::ofstream file;
			std::stringstream ss;
			ss << "../bin/time/processtime.txt";
			file.open(ss.str());
			ss.str("");
			auto AllData = ProcessingTime.Get();

			{
				auto data = ProcessingTime.Get("tracking");
				for (auto iter = data.begin(), iend = data.end(); iter != iend; iter++) {
					auto temp = iter->second;
					temp->update();
					ss << temp->print() << std::endl;
				}
			}
			{
				auto data = ProcessingTime.Get("mapping");
				for (auto iter = data.begin(), iend = data.end(); iter != iend; iter++) {
					auto temp = iter->second;
					temp->update();
					ss << temp->print() << std::endl;
				}
			}
			{
				auto data = ProcessingTime.Get("reloc");
				for (auto iter = data.begin(), iend = data.end(); iter != iend; iter++) {
					auto temp = iter->second;
					temp->update();
					ss << temp->print() << std::endl;
				}
			}
			{
				auto data = ProcessingTime.Get("download");
				for (auto iter = data.begin(), iend = data.end(); iter != iend; iter++) {
					auto temp = iter->second;
					temp->update();
					ss << temp->print() << std::endl;
				}
			}
			{
				auto data = ProcessingTime.Get("upload");
				for (auto iter = data.begin(), iend = data.end(); iter != iend; iter++) {
					auto temp = iter->second;
					temp->update();
					ss << temp->print() << std::endl;
				}
			}

			/*for (auto iter = AllData.begin(), iend = AllData.end(); iter != iend; iter++) {
				auto vec = iter->second;
				for (auto jter = vec.begin(), jend = vec.end(); jter != jend; jter++) {
					auto temp = jter->second;
					temp->update();
					ss << temp->print() << std::endl;
				}
			}*/
			file.write(ss.str().c_str(), ss.str().size());
			file.close();
		}
		{
			std::ofstream file;
			std::stringstream ss;
			ss << "../bin/time/ratio.txt";
			file.open(ss.str());
			ss.str("");
			{
				auto data = SuccessRatio.Get("async");
				for (auto iter = data.begin(), iend = data.end(); iter != iend; iter++) {
					auto temp = iter->second;
					temp->update();
					ss << temp->print() << std::endl;
				}
			}
			{
				auto data = SuccessRatio.Get("skipframe");
				for (auto iter = data.begin(), iend = data.end(); iter != iend; iter++) {
					auto temp = iter->second;
					temp->update();
					ss << temp->print() << std::endl;
				}
			}
			
			/*auto RatioData = SuccessRatio.Get();
			for (auto iter = RatioData.begin(), iend = RatioData.end(); iter != iend; iter++) {
				auto vec = iter->second;
				for (auto jter = vec.begin(), jend = vec.end(); jter != jend; jter++) {
					auto temp = jter->second;
					temp->update();
					ss << temp->print() << std::endl;
				}
			}*/
			file.write(ss.str().c_str(), ss.str().size());
			file.close();
		}
	}
	void SLAM::LoadProcessingTime(){
		
		/*for (int i = 1; i < 9; i++) {
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
		}*/
		{
			std::ifstream file;
			std::stringstream ss;
			ss << "../bin/time/processtime.txt";
			file.open(ss.str());
			ss.str("");
			std::string s;

			std::map<int, ProcessTime*> vec1;
			for (int i = 1; i < 9; i++)
			{
				auto p = new ProcessTime();
				getline(file, s);
				p->load(s);
				vec1.insert(std::make_pair(i, p));
			}
			ProcessingTime.Update("tracking", vec1);

			std::map<int, ProcessTime*> vec2;
			for (int i = 1; i < 9; i++)
			{
				auto p = new ProcessTime();
				getline(file, s);
				p->load(s);
				vec2.insert(std::make_pair(i, p));
			}
			ProcessingTime.Update("mapping", vec2);

			std::map<int, ProcessTime*> vec3;
			for (int i = 1; i < 9; i++)
			{
				auto p = new ProcessTime();
				getline(file, s);
				p->load(s);
				vec3.insert(std::make_pair(i, p));
			}
			ProcessingTime.Update("reloc", vec3);

			std::map<int, ProcessTime*> vec4;
			for (int i = 1; i < 9; i++)
			{
				auto p = new ProcessTime();
				getline(file, s);
				p->load(s);
				vec4.insert(std::make_pair(i, p));
			}
			ProcessingTime.Update("download", vec4);

			std::map<int, ProcessTime*> vec5;
			for (int i = 1; i < 9; i++)
			{
				auto p = new ProcessTime();
				getline(file, s);
				p->load(s);
				vec5.insert(std::make_pair(i, p));
			}
			ProcessingTime.Update("upload", vec5);
		}
		{
			std::ifstream file;
			std::stringstream ss;
			ss << "../bin/time/ratio.txt";
			file.open(ss.str());
			ss.str("");
			std::string s;

			std::map<int, Ratio*> vec1;
			for (int i = 10; i < 70; i+=10) {
				auto ratio = new Ratio();
				getline(file, s);
				//ss << s;
				ratio->load(s);
				vec1.insert(std::make_pair(i, ratio));
			}
			//for(int i = 10; i < )
			SuccessRatio.Update("async", vec1);

			std::map<int, Ratio*> vec;
			for (int i = 1; i < 15; i++) {
				//std::stringstream ss;
				/*ss << "../bin/time/skipframe_" << i << ".txt";*/
				auto ratio = new Ratio();
				getline(file, s);
				//ss << s;
				ratio->load(s);
				vec.insert(std::make_pair(i, ratio));
			}
			//for(int i = 10; i < )
			SuccessRatio.Update("skipframe", vec);
		}
		
	}
	void SLAM::SaveTrajectory(std::string user) {
		auto pUser = this->GetUser(user);
		if (!pUser)
			return;
		if (!pUser->mbSaveTrajectory)
			return;
		pUser->mnUsed++;

		std::stringstream ssPath;
		ssPath << "../bin/trajectory/" << pUser->mapName;
		
		int resPath = _access(ssPath.str().c_str(), 0);
		if (resPath == -1)
			_mkdir(ssPath.str().c_str());

		

		int nMapQuality = MapQuality.Get(pUser->mapName);
		int nUserQuality = pUser->mnQuality;

		if (pUser->mbAsyncTest) {
			ssPath << "/" << pUser->mapName << "_TrackingTest_" << pUser->mnQuality;
		}
		else {
			ssPath << "/" << pUser->mapName << "_" << pUser->mnQuality;
		}
		/*if (nMapQuality != nUserQuality && user->mbDeviceTracking) {
			ssPath << "/" << user->mapName << "_TrackingTest_" << user->mnQuality;
		}
		else {
			ssPath << "/" << user->mapName << "_" << user->mnQuality;
		}*/
		
		resPath = _access(ssPath.str().c_str(), 0);
		if (resPath == -1)
			_mkdir(ssPath.str().c_str());

		__time64_t long_time;
		_time64(&long_time);
		struct tm newtime;
		_localtime64_s(&newtime, &long_time);

		auto pMap = GetMap(pUser->mapName);
		auto vpKFs =  pMap->GetAllKeyFrames();

		std::stringstream ss;
		ss <<ssPath.str()<< "/" << pUser->userName<<"_"<< pUser->mnSkip<<(pUser->mbMapping?("_MAPPING_"):("_TRACKING_"))<< newtime.tm_year + 1900 <<"_"<< newtime.tm_mon+1<<"_"<< newtime.tm_mday<<"_"<< newtime.tm_hour<<"_"<< newtime.tm_min<<"_"<<newtime.tm_sec<< ".txt";
		std::ofstream f;
		f.open(ss.str().c_str());
		f << std::fixed;
		
		if (pUser->mbMapping) {
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
			for (int i = 0; i < pUser->vecTrajectories.size(); i += 2) {
				cv::Mat R = pUser->vecTrajectories[i].rowRange(0, 3).colRange(0, 3);
				cv::Mat t = pUser->vecTrajectories[i].rowRange(0, 3).col(3);
				R = R.t(); //inverse
				t = -R*t;  //camera center
				std::vector<float> q = Converter::toQuaternion(R);
				f << std::setprecision(6) << pUser->vecTimestamps[i] << std::setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
					<< " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
			}
		}
		f.close();

		if (pUser->mbDeviceTracking) {
			std::stringstream ss;
			ss << ssPath.str() << "/" << pUser->userName << "_" << pUser->mnSkip << "_DEVICE_" << newtime.tm_year + 1900 << "_" << newtime.tm_mon + 1 << "_" << newtime.tm_mday << "_" << newtime.tm_hour << "_" << newtime.tm_min << "_" << newtime.tm_sec << ".txt";
			std::ofstream f;
			f.open(ss.str().c_str());
			f << std::fixed;

			auto vecTrajectories = pUser->mvDeviceTrajectories.get();
			auto vecTimestamps = pUser->mvDeviceTimeStamps.get();
			
			for (int i = 0; i < vecTrajectories.size(); i+=10) {
				
				cv::Mat R = vecTrajectories[i].rowRange(0, 3);
				cv::Mat t = vecTrajectories[i].row(3).t();
				R = R.t(); //inverse
				t = -R*t;  //camera center
				std::vector<float> q = Converter::toQuaternion(R);
				f << std::setprecision(6) << vecTimestamps[i] << std::setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
					<< " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << std::endl;
			}
			f.close();
		}
		pUser->mnUsed--;
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
