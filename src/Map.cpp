#include <Map.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <KeyFrameDB.h>
#include <User.h>
#include <windows.h>

namespace EdgeSLAM {
	Map::Map(DBoW3::Vocabulary* voc, bool bFixScale):mnMaxKFid(0), mnBigChangeIdx(0), mnNumMappingFrames(0), mnNumLoopClosingFrames(0), mnNumPlaneEstimation(0), mnNextKeyFrameID(0), mnNextMapPointID(0), mState(MapState::NoImages), mpVoc(voc),
		mbResetRequested(false), mbFinishRequested(false), mbFinished(true),
		mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false),
		mpMatchedKF(nullptr), mnLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
		mbStopGBA(false), mpThreadGBA(nullptr), mbFixScale(bFixScale), mnFullBAIdx(0)
	{
		mpKeyFrameDB = new KeyFrameDB(voc);
		mvvPlanarMPs = std::vector<std::vector<cv::Mat>>(3);
	}
	/*Map::Map(ORBVocabulary* voc) : mnNumMappingFrames(0), mnNextKeyFrameID(0), mnNextMapPointID(0), mState(MapState::NoImages) {
		mpKeyFrameDB = new KeyFrameDB(voc);
	}*/
	Map::~Map() {

	}
	
	MapState Map::GetState() {
		std::unique_lock<std::mutex> lock(mMutexState);
		return mState;
	}
	void Map::SetState(MapState stat) {
		std::unique_lock<std::mutex> lock(mMutexState);
		mState = stat;
	}

	void Map::AddMapPoint(MapPoint* pMP){
		std::unique_lock<std::mutex> lock(mMutexMPs);
		if (!mspMapPoints.count(pMP))
			mspMapPoints.insert(pMP);
	}
	void Map::RemoveMapPoint(MapPoint* pMP){
		std::unique_lock<std::mutex> lock(mMutexMPs);
		if (mspMapPoints.count(pMP))
			mspMapPoints.erase(pMP);
	}
	std::vector<MapPoint*> Map::GetAllMapPoints(){
		std::unique_lock<std::mutex> lock(mMutexMPs);
		return std::vector<MapPoint*>(mspMapPoints.begin(), mspMapPoints.end());
	}
	int Map::GetNumMapPoints(){
		std::unique_lock<std::mutex> lock(mMutexMPs);
		return mspMapPoints.size();
	}

	void Map::AddKeyFrame(KeyFrame* pF){
		int id = pF->mnId;
		std::unique_lock<std::mutex> lock(mMutexKFs);
		if (!mmpKeyFrames.count(id)) {
			mmpKeyFrames[id] = pF;
			/*pF->mnKeyFrameID = nServerKeyFrameID++;
			mpPrevKF = mpCurrKF;
			mpCurrKF = pF;*/
		}
		if (pF->mnId > mnMaxKFid)
			mnMaxKFid = pF->mnId;
	}
	KeyFrame* Map::GetKeyFrame(int id) {
		std::unique_lock<std::mutex> lock(mMutexKFs);
		if (mmpKeyFrames.count(id))
			return mmpKeyFrames[id];
		return nullptr;
	}
	void Map::RemoveKeyFrame(KeyFrame* pF){
		int id = pF->mnId;
		std::unique_lock<std::mutex> lock(mMutexKFs);
		if (mmpKeyFrames.count(id))
			mmpKeyFrames.erase(id);
	}
	std::vector<KeyFrame*> Map::GetAllKeyFrames(){
		std::unique_lock<std::mutex> lock(mMutexKFs);
		std::vector<KeyFrame*> res;
		for (auto iter = mmpKeyFrames.begin(), iend = mmpKeyFrames.end(); iter != iend; iter++)
		{
			auto pKF = iter->second;
			res.push_back(pKF);
		}
		return std::vector<KeyFrame*>(res.begin(), res.end());
	}
	int Map::GetNumKeyFrames(){
		std::unique_lock<std::mutex> lock(mMutexKFs);
		return mmpKeyFrames.size();
	}
	void Map::Delete(){
		{
			{
				std::unique_lock<std::mutex> lock(mMutexMPs);
				for (auto iter = mmpKeyFrames.begin(), iend = mmpKeyFrames.end(); iter != iend; iter++) {
					delete iter->second;
				}
			}
			{
				std::unique_lock<std::mutex> lock(mMutexKFs);
				for (auto iter = mmpKeyFrames.begin(), iend = mmpKeyFrames.end(); iter != iend; iter++)
				{
					auto pKF = iter->second;
					pKF->reset_map_points();
					delete pKF;
				}
			}

			mvpKeyFrameOrigins.clear();
			mnNextKeyFrameID = 0;
			mnNextMapPointID = 0;
			mlpNewMPs.clear();
			mState = MapState::NoImages;

			//add variable
		}
	}

	void Map::InformNewBigChange()
	{
		mnBigChangeIdx++;
	}

	int Map::GetLastBigChangeIdx()
	{
		return mnBigChangeIdx.load();
	}
	/////Planar Test
	std::vector<cv::Mat> Map::GetPlanarMPs(int id){
		std::unique_lock<std::mutex> lock(mMutexPlanarMP);
		return mvvPlanarMPs[id];
	}
	void Map::ClearPlanarMPs(){
		std::unique_lock<std::mutex> lock(mMutexPlanarMP);
		for (int i = 0, iend = mvvPlanarMPs.size(); i < iend; i++)
			mvvPlanarMPs[i].clear();
	}
	void Map::AddPlanarMP(cv::Mat m, int id){
		std::unique_lock<std::mutex> lock(mMutexPlanarMP);
		mvvPlanarMPs[id].push_back(m);
	}
	/////Planar Test
	/////Depth test
	std::vector<cv::Mat> Map::GetDepthMPs() {
		std::unique_lock<std::mutex> lock(mMutexDepthTest);
		return std::vector<cv::Mat>(mvTempMPs.begin(), mvTempMPs.end());
	}
	void Map::ClearDepthMPs() {
		std::unique_lock<std::mutex> lock(mMutexDepthTest);
		mvTempMPs.clear();
	}
	void Map::AddDepthMP(cv::Mat m) {
		std::unique_lock<std::mutex> lock(mMutexDepthTest);
		mvTempMPs.push_back(m);
	}
	/////Depth test

	//////////////Thread sync
	void Map::RequestStop()
	{
		mbStopRequested = true;
		mbAbortBA = true;
	}
	bool Map::Stop()
	{
		if (mbStopRequested && !mbNotStop)
		{ 
			mbStopped = true;
			return true;
		}
		return false;
	}
	bool Map::isStopped()
	{
		return mbStopped;
	}
	bool Map::stopRequested()
	{
		return mbStopRequested.load();
	}
	bool Map::SetNotStop(bool flag)
	{

		if (flag && mbStopped)
			return false;
		mbNotStop = flag;
		return true;
	}
	void Map::Release()
	{
		if (mbFinished)
			return;
		mbStopped = false;
		mbStopRequested = false;
		/*for (std::list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend = mlNewKeyFrames.end(); lit != lend; lit++)
		delete *lit;
		mlNewKeyFrames.clear();*/
	}
	void Map::RequestReset()
	{
		{
			mbResetRequested = true;
		}

		while (1)
		{
			{
				if (!mbResetRequested)
					break;
			}
			Sleep(3000);
		}
	}

	void Map::ResetIfRequested()
	{
		if (mbResetRequested)
		{
			//mlNewKeyFrames.clear();
			//mlpRecentAddedMapPoints.clear();
			mbResetRequested = false;
		}
	}

	void Map::RequestFinish()
	{
		mbFinishRequested = true;
	}

	bool Map::CheckFinish()
	{
		return mbFinishRequested.load();
	}

	void Map::SetFinish()
	{
		mbFinished = true;
		mbStopped = true;
	}

	bool Map::isFinished()
	{
		return mbFinished.load();
	}

	bool Map::isRunningGBA(){
		return mbRunningGBA.load();
	}
	bool Map::isFinishedGBA(){
		return mbFinishedGBA.load();
	}

	//////////////Thread sync

	////////////////Local Map
	LocalMap::LocalMap(){}
	LocalMap::~LocalMap(){}
	LocalCovisibilityMap::LocalCovisibilityMap() :LocalMap()
	{}
	LocalCovisibilityMap::~LocalCovisibilityMap() {

	}
	/////////////keyframe
	void LocalCovisibilityMap::UpdateLocalMap(KeyFrame* kf, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs){
		UpdateLocalKeyFrames(kf, vpLocalKFs);
		UpdateLocalMapPoitns(kf, vpLocalKFs, vpLocalMPs);
	}
	void LocalCovisibilityMap::UpdateLocalKeyFrames(KeyFrame* kf, std::vector<KeyFrame*>& vpLocalKFs){
		// Each map point vote for the keyframes in which it has been observed
		std::map<KeyFrame*, int> keyframeCounter;
		int nFrameID = kf->mnId;
		for (int i = 0; i<kf->N; i++)
		{
			auto pMP = kf->mvpMapPoints.get(i);
			if (pMP && !pMP->isBad()) {
				const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();
				for (std::map<KeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
					keyframeCounter[it->first]++;
			}
			else {
				kf->mvpMapPoints.update(i,nullptr);
			}
		}

		if (keyframeCounter.empty())
			return;

		int max = 0;
		KeyFrame* pKFmax = static_cast<KeyFrame*>(nullptr);
		std::set<KeyFrame*> spKFs;

		// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
		for (std::map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
		{
			KeyFrame* pKF = it->first;

			if (pKF->isBad())
				continue;

			if (it->second>max)
			{
				max = it->second;
				pKFmax = pKF;
			}

			vpLocalKFs.push_back(it->first);
			spKFs.insert(pKF);
		}

		// Include also some not-already-included keyframes that are neighbors to already-included keyframes
		//for (std::vector<KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		for (size_t i = 0, iend = vpLocalKFs.size(); i < iend; i++)
		{
			// Limit the number of keyframes
			if (vpLocalKFs.size()>80)
				break;

			KeyFrame* pKF = vpLocalKFs[i];// *itKF;

			const std::vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

			for (std::vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
			{
				KeyFrame* pNeighKF = *itNeighKF;
				if (pNeighKF && !pNeighKF->isBad() && !spKFs.count(pNeighKF))
				{
					spKFs.insert(pNeighKF);
					break;
				}
			}

			const std::set<KeyFrame*> spChilds = pKF->GetChilds();
			for (std::set<KeyFrame*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
			{
				KeyFrame* pChildKF = *sit;
				if (pChildKF && !pChildKF->isBad() && !spKFs.count(pChildKF))
				{
					spKFs.insert(pChildKF);
					break;
				}
			}

			KeyFrame* pParent = pKF->GetParent();
			if (pParent && !pParent->isBad() && !spKFs.count(pParent))
			{
				spKFs.insert(pParent);
				break;
			}

		}

	}
	void LocalCovisibilityMap::UpdateLocalMapPoitns(KeyFrame* kf, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs){
		std::set<MapPoint*> spMPs;
		for (std::vector<KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			KeyFrame* pKF = *itKF;
			const std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

			for (std::vector<MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
				MapPoint* pMP = *itMP;
				if (!pMP || pMP->isBad() || spMPs.count(pMP))
					continue;
				vpLocalMPs.push_back(pMP);
				spMPs.insert(pMP);
			}
		}
	}
	////////////keyframe

	void LocalCovisibilityMap::UpdateLocalMap(User* user, Frame* f) {
		UpdateLocalKeyFrames(user, f, mvpLocalKFs);
		UpdateLocalMapPoitns(f, mvpLocalKFs, mvpLocalMPs, mvpLocalTPs);
	}
	void LocalCovisibilityMap::UpdateLocalKeyFrames(User* user, Frame* f, std::vector<KeyFrame*>& vpLocalKFs) {
		// Each map point vote for the keyframes in which it has been observed
		std::map<KeyFrame*, int> keyframeCounter;
		int nFrameID = f->mnFrameID;
		for (int i = 0; i<f->N; i++)
		{
			auto pMP = f->mvpMapPoints[i];
			if (pMP && !pMP->isBad()) {
				const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();
				for (std::map<KeyFrame*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
					keyframeCounter[it->first]++;
			}
			else {
				f->mvpMapPoints[i] = nullptr;
			}
		}
		
		if (keyframeCounter.empty())
			return;

		int max = 0;
		KeyFrame* pKFmax = static_cast<KeyFrame*>(nullptr);
		std::set<KeyFrame*> spKFs;

		// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
		for (std::map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
		{
			KeyFrame* pKF = it->first;

			if (pKF->isBad())
				continue;

			if (it->second>max)
			{
				max = it->second;
				pKFmax = pKF;
			}

			vpLocalKFs.push_back(it->first);
			spKFs.insert(pKF);
		}

		// Include also some not-already-included keyframes that are neighbors to already-included keyframes
		//for (std::vector<KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		for(size_t i =0, iend = vpLocalKFs.size(); i < iend; i++)
		{
			// Limit the number of keyframes
			if (vpLocalKFs.size()>80)
				break;

			KeyFrame* pKF = vpLocalKFs[i];// *itKF;

			const std::vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

			for (std::vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
			{
				KeyFrame* pNeighKF = *itNeighKF;
				if (pNeighKF && !pNeighKF->isBad() && !spKFs.count(pNeighKF))
				{
					spKFs.insert(pNeighKF);
					break;
					/*if (pNeighKF->mnTrackReferenceForFrame != nFrameID)
					{
						vpLocalKFs.push_back(pNeighKF);
						pNeighKF->mnTrackReferenceForFrame = nFrameID;
						break;
					}*/
				}
			}

			const std::set<KeyFrame*> spChilds = pKF->GetChilds();
			for (std::set<KeyFrame*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
			{
				KeyFrame* pChildKF = *sit;
				if (pChildKF && !pChildKF->isBad() && !spKFs.count(pChildKF))
				{
					spKFs.insert(pChildKF);
					break;
					/*if (pChildKF->mnTrackReferenceForFrame != nFrameID)
					{
						vpLocalKFs.push_back(pChildKF);
						pChildKF->mnTrackReferenceForFrame = nFrameID;
						break;
					}*/
				}
			}

			KeyFrame* pParent = pKF->GetParent();
			if (pParent && !pParent->isBad() && !spKFs.count(pParent))
			{
				spKFs.insert(pParent);
				break;
				/*if (pParent->mnTrackReferenceForFrame != nFrameID)
				{
					vpLocalKFs.push_back(pParent);
					pParent->mnTrackReferenceForFrame = nFrameID;
					break;
				}*/
			}

		}
		if (pKFmax)
		{
			user->mnReferenceKeyFrameID = pKFmax->mnId;
		}
	}
	void LocalCovisibilityMap::UpdateLocalMapPoitns(Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs, std::vector<TrackPoint*>& vpLocalTPs){
		std::set<MapPoint*> spMPs;
		for (std::vector<KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			KeyFrame* pKF = *itKF;
			const std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

			for (std::vector<MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
				MapPoint* pMP = *itMP;
				if (!pMP || pMP->isBad() || spMPs.count(pMP))
					continue;
				vpLocalMPs.push_back(pMP);
				vpLocalTPs.push_back(new TrackPoint());
				spMPs.insert(pMP);
				//pMP->mnTrackReferenceForFrame = f->mnFrameID;
			}
		}
	}
}