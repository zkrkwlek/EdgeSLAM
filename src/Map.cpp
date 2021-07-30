#include <Map.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <KeyFrameDB.h>
#include <User.h>
#include <windows.h>

namespace EdgeSLAM {
	Map::Map(DBoW3::Vocabulary* voc, bool bFixScale):mnMaxKFid(0), mnBigChangeIdx(0), mnNumMappingFrames(0), mnNumLoopClosingFrames(0), mnNextKeyFrameID(0), mnNextMapPointID(0), mState(MapState::NoImages), mpVoc(voc),
		mbResetRequested(false), mbFinishRequested(false), mbFinished(true),
		mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false),
		mpMatchedKF(nullptr), mnLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
		mbStopGBA(false), mpThreadGBA(nullptr), mbFixScale(bFixScale), mnFullBAIdx(0)
	{
		mpKeyFrameDB = new KeyFrameDB(voc);
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
		std::unique_lock<std::mutex> lock(mMutexKFs);
		if (!mspKeyFrames.count(pF)) {
			mspKeyFrames.insert(pF);
			/*pF->mnKeyFrameID = nServerKeyFrameID++;
			mpPrevKF = mpCurrKF;
			mpCurrKF = pF;*/
		}
		if (pF->mnId > mnMaxKFid)
			mnMaxKFid = pF->mnId;
	}
	void Map::RemoveKeyFrame(KeyFrame* pF){
		std::unique_lock<std::mutex> lock(mMutexKFs);
		if (mspKeyFrames.count(pF))
			mspKeyFrames.erase(pF);
	}
	std::vector<KeyFrame*> Map::GetAllKeyFrames(){
		std::unique_lock<std::mutex> lock(mMutexKFs);
		return std::vector<KeyFrame*>(mspKeyFrames.begin(), mspKeyFrames.end());
	}
	int Map::GetNumKeyFrames(){
		std::unique_lock<std::mutex> lock(mMutexKFs);
		return mspKeyFrames.size();
	}
	void Map::Delete(){
		{
			{
				std::unique_lock<std::mutex> lock(mMutexMPs);
				for (auto iter = mspKeyFrames.begin(), iend = mspKeyFrames.end(); iter != iend; iter++) {
					delete *iter;
				}
			}
			{
				std::unique_lock<std::mutex> lock(mMutexKFs);
				for (auto iter = mspKeyFrames.begin(), iend = mspKeyFrames.end(); iter != iend; iter++)
				{
					auto pKF = *iter;
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
	void LocalCovisibilityMap::UpdateLocalMap(User* user, Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs) {
		UpdateLocalKeyFrames(user, f, vpLocalKFs);
		//std::cout << "UpdateKF::endl" << std::endl;
		UpdateLocalMapPoitns(f, vpLocalKFs, vpLocalMPs);
		//std::cout << "UpdateMP::endl" << std::endl;
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
		//std::cout << "2" << std::endl;
		if (keyframeCounter.empty())
			return;

		int max = 0;
		KeyFrame* pKFmax = static_cast<KeyFrame*>(nullptr);
		
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
			pKF->mnTrackReferenceForFrame = nFrameID;
		}
		//std::cout << "3" << std::endl;

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
				if (pNeighKF && !pNeighKF->isBad())
				{
					if (pNeighKF->mnTrackReferenceForFrame != nFrameID)
					{
						vpLocalKFs.push_back(pNeighKF);
						pNeighKF->mnTrackReferenceForFrame = nFrameID;
						break;
					}
				}
			}

			const std::set<KeyFrame*> spChilds = pKF->GetChilds();
			for (std::set<KeyFrame*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
			{
				KeyFrame* pChildKF = *sit;
				if (pChildKF && !pChildKF->isBad())
				{
					if (pChildKF->mnTrackReferenceForFrame != nFrameID)
					{
						vpLocalKFs.push_back(pChildKF);
						pChildKF->mnTrackReferenceForFrame = nFrameID;
						break;
					}
				}
			}

			KeyFrame* pParent = pKF->GetParent();
			if (pParent && !pParent->isBad())
			{
				if (pParent->mnTrackReferenceForFrame != nFrameID)
				{
					vpLocalKFs.push_back(pParent);
					pParent->mnTrackReferenceForFrame = nFrameID;
					break;
				}
			}

		}
		//std::cout << "4" << std::endl;
		if (pKFmax)
		{
			user->mnReferenceKeyFrameID = pKFmax->mnId;
		}
		//std::cout << "5" << std::endl;
	}
	void LocalCovisibilityMap::UpdateLocalMapPoitns(Frame* f, std::vector<KeyFrame*>& vpLocalKFs, std::vector<MapPoint*>& vpLocalMPs){
		for (std::vector<KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		{
			KeyFrame* pKF = *itKF;
			const std::vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

			for (std::vector<MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
				MapPoint* pMP = *itMP;
				if (!pMP || pMP->isBad() || pMP->mnTrackReferenceForFrame == f->mnFrameID)
					continue;
				vpLocalMPs.push_back(pMP);
				pMP->mnTrackReferenceForFrame = f->mnFrameID;
			}
		}
	}
}