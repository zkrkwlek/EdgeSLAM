#include <ObjectFrame.h>
#include <Converter.h>
#include <FeatureTracker.h>
#include <KeyFrame.h>
#include <Frame.h>
//#include <ConcurrentMap.h>
//#include <MapPoint.h>

namespace EdgeSLAM {

	std::atomic<int> nNextBBoxID = 0;
	std::atomic<int> nNextObjNodeID = 0;

	ObjectTrackingFrame::ObjectTrackingFrame(){}
	ObjectTrackingFrame::~ObjectTrackingFrame(){
		frame.release();
		std::vector<MapPoint*>().swap(mvpMapPoints);
		std::vector<cv::Point2f>().swap(mvImagePoints);
	}

	ObjectTrackingResult::ObjectTrackingResult(ObjectNode* _pObj, int _label, std::string _user):
		mState(ObjectTrackingState::NotEstimated), mnLastSuccessFrameId(-1), mnLastTrackFrameId(-1), Pose(cv::Mat::eye(4,4,CV_32FC1)),
		mpObject(_pObj), mnObjectLabelId(_label), mStrDeviceName(_user), mpLastFrame(nullptr)
	{}
	ObjectTrackingResult::~ObjectTrackingResult(){
		if (mpLastFrame)
			delete mpLastFrame;
	}

	ObjectLocalMap::ObjectLocalMap(std::vector<MapPoint*> vpMPs){
		std::map<ObjectBoundingBox*, int> boxCounter;
		std::set<ObjectBoundingBox*> spLocalBoxes;

		for (int i = 0, iend = vpMPs.size(); i < iend; i++)
		{
			auto pMP = vpMPs[i];
			if (!pMP || pMP->isBad())
				continue;
			auto pOP = pMP->mpObjectPoint;
			const std::map<ObjectBoundingBox*, size_t> observations = pOP->GetObservations();
			for (std::map<ObjectBoundingBox*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
				boxCounter[it->first]++;
			
		}

		if (boxCounter.empty())
			return;
		int max = 0;
		ObjectBoundingBox* pBOXmax = nullptr;

		for (std::map<ObjectBoundingBox*, int>::const_iterator it = boxCounter.begin(), itEnd = boxCounter.end(); it != itEnd; it++)
		{
			ObjectBoundingBox* pBox = it->first;

			if (it->second > max)
			{
				max = it->second;
				pBOXmax = pBox;
			}

			mvpLocalBoxes.push_back(it->first);
			spLocalBoxes.insert(pBox);
		}

		std::set<MapPoint*> spMPs;
		for (std::vector<ObjectBoundingBox*>::const_iterator itKF = mvpLocalBoxes.begin(), itEndKF = mvpLocalBoxes.end(); itKF != itEndKF; itKF++)
		{
			if (spMPs.size() >= 4000)
				break;
			ObjectBoundingBox* pBox = *itKF;
			const std::vector<MapPoint*> vpMPs = pBox->mvpMapPoints.get();

			for (std::vector<MapPoint*>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
			{
				MapPoint* pMP = *itMP;
				if (!pMP || pMP->isBad() || spMPs.count(pMP))
					continue;
				mvpLocalMapPoints.push_back(pMP);
				spMPs.insert(pMP);
			}
		}
	}
	ObjectLocalMap::~ObjectLocalMap(){}

	ObjectMapPoint::ObjectMapPoint(ObjectBoundingBox* pRefBB, MapPoint* pMP):mpRefBoudingBox(pRefBB),matPosObject(cv::Mat::zeros(3,1,CV_32FC1)), mpMapPoint(pMP), mpObjectMap(nullptr), nObs(0)
	{
		//this->SetObjectPos(this->GetWorldPos());
	}
	ObjectMapPoint::~ObjectMapPoint(){}
	void ObjectMapPoint::AddObservation(ObjectBoundingBox* pBB, size_t idx){
		if (mObservations.Count(pBB))
			return;
		mObservations.Update(pBB, idx);
		nObs++;
	}

	
	void ObjectMapPoint::EraseObservation(ObjectBoundingBox* pBB){
		//레퍼런스 변경 등의 수정이 필요함.
		bool bBad = false;
		if (mObservations.Count(pBB)) {
			nObs--;
			mObservations.Erase(pBB);

			//레퍼런스 바운딩 박스의 변경

			if (nObs <= 2)
				bBad = true;
		}
		if(bBad){
			//수정 필요함
			//SetBadFlag();
		}
	}

	
	int ObjectMapPoint::GetIndexInKeyFrame(ObjectBoundingBox* pBox) {
		if (mObservations.Count(pBox))
			return mObservations.Get(pBox);
		return -1;
	}

	std::map<ObjectBoundingBox*, size_t> ObjectMapPoint::GetObservations()
	{
		return mObservations.Get();
	}
	/*
	void ObjectMapPoint::SetBadFlag() {
		auto obs = mObservations.Get();
		for (std::map<ObjectBoundingBox*, size_t>::iterator mit = obs.begin(), mend = obs.end(); mit != mend; mit++)
		{
			ObjectBoundingBox* pKF = mit->first;
			pKF->EraseObjectPointMatch(mit->second);
		}
		mpObjectMap->RemoveMapPoint(this);
	}

	cv::Mat ObjectMapPoint::GetObjectPos(){
		std::unique_lock<std::mutex> lock(mMutexObjectPos);
		return matPosObject.clone();
	}

	void ObjectMapPoint::SetObjectPos(const cv::Mat& _pos){
		std::unique_lock<std::mutex> lock(mMutexObjectPos);
		matPosObject = _pos.clone();
	}

	void ObjectMapPoint::ComputeDistinctiveDescriptors() {
		std::vector<cv::Mat> vDescriptors;
		std::map<ObjectBoundingBox*, size_t> observations = mObservations.Get();

		if (observations.empty())
			return;

		for (std::map<ObjectBoundingBox*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
		{
			auto pBB = mit->first;

			//if (!pKF->isBad())
			vDescriptors.push_back(pBB->desc.row(mit->second));
		}

		if (vDescriptors.empty())
			return;

		// Compute distances between them
		size_t N = vDescriptors.size();
		std::vector<std::vector<float> > Distances;
		Distances.resize(N, std::vector<float>(N, 0));
		for (size_t i = 0; i < N; i++)
		{
			Distances[i][i] = 0;
			for (size_t j = i + 1; j < N; j++)
			{
				int distij = (int)mpDist->DescriptorDistance(vDescriptors[i], vDescriptors[j]);//ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
				Distances[i][j] = distij;
				Distances[j][i] = distij;
			}
		}


		// Take the descriptor with least median distance to the rest
		int BestMedian = INT_MAX;
		int BestIdx = 0;
		for (size_t i = 0; i < N; i++)
		{
			std::vector<int> vDists(Distances[i].begin(), Distances[i].end());
			sort(vDists.begin(), vDists.end());
			int median = vDists[0.5 * (N - 1)];

			if (median < BestMedian)
			{
				BestMedian = median;
				BestIdx = i;
			}
		}

		{
			std::unique_lock<std::mutex> lock(mMutexFeatures);
			mDescriptor = vDescriptors[BestIdx].clone();
		}

	}
	*/

	ObjectNode::ObjectNode():mnId(++nNextObjNodeID), origin(cv::Mat::zeros(3, 1, CV_32FC1)),radius(0.0), desc(cv::Mat::zeros(0, 32, CV_8UC1)), matWorldPose(cv::Mat::eye(4, 4, CV_32FC1)), matObjPose(cv::Mat::eye(4, 4, CV_32FC1)) {
	}
	ObjectNode::~ObjectNode() {
		origin.release();
		desc.release();
		matWorldPose.release();
		matObjPose.release();
		mspOPs.Release();
		mspKFs.Release();
		mspBBs.Release();
	}
	void ObjectNode::ComputeBow(DBoW3::Vocabulary* voc) {
		if (mBowVec.empty())
		{
			std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(desc);
			voc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);  // 5 is better
		}
	}

	void ObjectNode::ClearDescriptor() {
		std::unique_lock<std::mutex> lock(mMutexDesc);
		desc = cv::Mat::zeros(0, 32, CV_8UC1);
	}
	void ObjectNode::AddDescriptor(cv::Mat _row){
		std::unique_lock<std::mutex> lock(mMutexDesc);
		desc.push_back(_row);
	}
	cv::Mat ObjectNode::GetDescriptor() {
		std::unique_lock<std::mutex> lock(mMutexDesc);
		return desc.clone();
	}

	
	void ObjectNode::SetWorldPose(const cv::Mat& _pos){
		std::unique_lock<std::mutex> lock(mMutexWorldPose);
		matWorldPose = _pos.clone();
	}
	void ObjectNode::SetObjectPose(const cv::Mat& _pos){
		std::unique_lock<std::mutex> lock(mMutexObjPose);
		matObjPose = _pos.clone();
	}
	cv::Mat ObjectNode::GetWorldPose(){
		std::unique_lock<std::mutex> lock(mMutexWorldPose);
		return matWorldPose.clone();
	}
	cv::Mat ObjectNode::GetObjectPose(){
		std::unique_lock<std::mutex> lock(mMutexObjPose);
		return matObjPose.clone();
	}

	void ObjectNode::UpdateOrigin(){

		cv::Mat newOrigin = cv::Mat::zeros(3,1,CV_32FC1);
		float newRadius = 0.0;

		auto vecMPs = mspMPs.ConvertVector();
		int N = 0;
		for (int i = 0, iend = vecMPs.size(); i < iend; i++) {
			auto pOP = vecMPs[i];
			if(pOP && !pOP->isBad()){
				N++;
				newOrigin += pOP->GetWorldPos();
			}
		}
		newOrigin /= N;
		for (int i = 0, iend = vecMPs.size(); i < iend; i++) {
			auto pOP = vecMPs[i];
			if(pOP && !pOP->isBad()){
				cv::Mat temp = pOP->GetWorldPos() - newOrigin;
				newRadius+=sqrt(temp.dot(temp));
			}
		}

		{
			std::unique_lock<std::mutex> lock(mMutexOrigin);
			origin = newOrigin.clone();
			radius = newRadius/N;
		}
	}
	cv::Mat ObjectNode::GetOrigin(){
		std::unique_lock<std::mutex> lock(mMutexOrigin);
		return origin.clone();
	}

	void ObjectNode::UpdateObjectPos() {
		//오브젝트 좌표계 계산
		/*auto vecMPs = mspOPs.ConvertVector();
		cv::Mat tempOrigin = cv::Mat::zeros(3, 1, CV_32FC1);
		{
			std::unique_lock<std::mutex> lock(mMutexOrigin);
			tempOrigin = origin.clone();	
		}
		for (int i = 0, iend = vecMPs.size(); i < iend; i++) {
			auto pOP = vecMPs[i];
			if (pOP && !pOP->isBad()) {
				pOP->SetObjectPos(pOP->GetWorldPos() - tempOrigin);
			}
		}*/
	}

	void ObjectNode::RemoveMapPoint(ObjectMapPoint* pMP) {
		mspOPs.Erase(pMP);
	}
	

	//ObjectBoundingBox::ObjectBoundingBox():id(0),mpNode(nullptr), mpKF(nullptr){}
	/*ObjectBoundingBox::ObjectBoundingBox(int _label, float _conf, cv::Point2f pt1, cv::Point2f pt2) : id(++nNextBBoxID), mpKF(nullptr), label(_label), confidence(_conf), rect(cv::Rect(pt1, pt2)),
		desc(cv::Mat::zeros(0, 32, CV_8UC1)), mpNode(nullptr)
	{

	}*/
	ObjectBoundingBox::ObjectBoundingBox(Frame* _pF, int _label, float _conf, cv::Point2f pt1, cv::Point2f pt2): 
		id(++nNextBBoxID), mpKF(nullptr), mpF(_pF), label(_label), confidence(_conf), rect(cv::Rect(pt1, pt2)),
		desc(cv::Mat::zeros(0, 32, CV_8UC1)), mpNode(nullptr),
		fx(_pF->fx), fy(_pF->fy), cx(_pF->cx), cy(_pF->cy),
		mnScaleLevels(_pF->mnScaleLevels), mfScaleFactor(_pF->mfScaleFactor),
		mfLogScaleFactor(_pF->mfLogScaleFactor), mvScaleFactors(_pF->mvScaleFactors),
		mvLevelSigma2(_pF->mvLevelSigma2), mvInvLevelSigma2(_pF->mvInvLevelSigma2)
	{
		K = cv::Mat::eye(3, 3, CV_32FC1);
		K.at<float>(0, 0) = fx;
		K.at<float>(1, 1) = fy;
		K.at<float>(0, 2) = cx;
		K.at<float>(1, 2) = cy;
	}
	ObjectBoundingBox::ObjectBoundingBox(KeyFrame* _pKF, int _label, float _conf, cv::Point2f pt1, cv::Point2f pt2):
		id(++nNextBBoxID), mpKF(_pKF), mpF(nullptr),label(_label), confidence(_conf), rect(cv::Rect(pt1, pt2)),
		desc(cv::Mat::zeros(0,32,CV_8UC1)), mpNode(nullptr),
		fx(_pKF->fx), fy(_pKF->fy), cx(_pKF->cx), cy(_pKF->cy),
		mnScaleLevels(_pKF->mnScaleLevels), mfScaleFactor(_pKF->mfScaleFactor),
		mfLogScaleFactor(_pKF->mfLogScaleFactor), mvScaleFactors(_pKF->mvScaleFactors), 
		mvLevelSigma2(_pKF->mvLevelSigma2), mvInvLevelSigma2(_pKF->mvInvLevelSigma2)
	{
		K = cv::Mat::eye(3, 3, CV_32FC1);
		K.at<float>(0, 0) = fx;
		K.at<float>(1, 1) = fy;
		K.at<float>(0, 2) = cx;
		K.at<float>(1, 2) = cy;
	}
	ObjectBoundingBox::~ObjectBoundingBox(){}

	void ObjectBoundingBox::ComputeBow(DBoW3::Vocabulary* voc) {
		std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(desc);
		voc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
	}
	void ObjectBoundingBox::AddMapPoint(MapPoint* pMP, size_t idx) {
		mvpMapPoints.update(idx, pMP);
		mspMapPoints.Update(pMP);
		
		if (pMP) {
			auto pOP = pMP->mpObjectPoint;
			if (!pOP) {
				pOP = new ObjectMapPoint(this, pMP);
				pMP->mpObjectPoint = pOP;
			}
			pOP->AddObservation(this, idx);
		}
		
	}
	void ObjectBoundingBox::AddMapPoint(MapPoint* pMP) {
		int idx = mvpMapPoints.size();
		mvpMapPoints.push_back(pMP);
		mspMapPoints.Update(pMP);
		mvbOutliers.push_back(false);
		mvpObjectPoints.push_back(nullptr);
		if (pMP) {
			auto pOP = pMP->mpObjectPoint;
			if (!pOP) {
				pOP = new ObjectMapPoint(this, pMP);
				pMP->mpObjectPoint = pOP;
			}
			pOP->AddObservation(this, idx);
		}
	}
	void ObjectBoundingBox::EraseObjectPointMatch(ObjectMapPoint* pMP) {
		int idx = pMP->GetIndexInKeyFrame(this);
		if (idx >= 0)
			mvpObjectPoints.update(idx, nullptr);
	}

	void ObjectBoundingBox::EraseObjectPointMatch(const size_t& idx) {
		mvpObjectPoints.update(idx, nullptr);
	}

	void ObjectBoundingBox::AddConnection(ObjectBoundingBox* pBox, const int& weight){
		mConnectedBoxWeights.Update(pBox, weight);
		UpdateBestCovisibles();
	}
	void ObjectBoundingBox::EraseConnection(ObjectBoundingBox* pBox){
		if (mConnectedBoxWeights.Count(pBox)) {
			mConnectedBoxWeights.Erase(pBox);
			UpdateBestCovisibles();
		}
	}
	void ObjectBoundingBox::UpdateConnections(ObjectNode* Object) {
		//auto spBBs = Object->mspBBs.Get();
		std::map<ObjectBoundingBox*, int> BOXcounter;

		std::vector<MapPoint*> vpMP = mvpMapPoints.get();
		for (std::vector<MapPoint*>::iterator vit = vpMP.begin(), vend = vpMP.end(); vit != vend; vit++)
		{
			MapPoint* pMP = *vit;

			if (!pMP || pMP->isBad())
				continue;

			auto pOP = pMP->mpObjectPoint;
			if (!pOP)
				continue;

			std::map<ObjectBoundingBox*, size_t> observations = pOP->GetObservations();
			
			for (std::map<ObjectBoundingBox*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				if (mit->first->id == id)
					continue;
				BOXcounter[mit->first]++;
			}
		}
		
		if (BOXcounter.empty())
			return;
		Object->mspBBs.Update(this);
		this->mpNode = Object;
		//If the counter is greater than threshold add connection
		//In case no keyframe counter is over threshold add the one with maximum counter
		int nmax = 0;
		ObjectBoundingBox* pBOXmax = nullptr;
		int th = 10;

		std::vector<std::pair<int, ObjectBoundingBox*> > vPairs;
		//vPairs.reserve(KFcounter.size());
		for (std::map<ObjectBoundingBox*, int>::iterator mit = BOXcounter.begin(), mend = BOXcounter.end(); mit != mend; mit++)
		{
			if (mit->second > nmax)
			{
				nmax = mit->second;
				pBOXmax = mit->first;
			}
			if (mit->second >= th)
			{
				vPairs.push_back(std::make_pair(mit->second, mit->first));
				(mit->first)->AddConnection(this, mit->second);
			}
		}

		if (vPairs.empty())
		{
			vPairs.push_back(std::make_pair(nmax, pBOXmax));
			pBOXmax->AddConnection(this, nmax);
		}

		sort(vPairs.begin(), vPairs.end());
		std::list<ObjectBoundingBox*> lKFs;
		std::list<int> lWs;
		for (size_t i = 0; i < vPairs.size(); i++)
		{
			lKFs.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		mConnectedBoxWeights.Copy(BOXcounter);// = BOXcounter;
		auto tempBoxes = std::vector<ObjectBoundingBox*>(lKFs.begin(), lKFs.end());
		mvpOrderedConnectedBoxes.Copy(tempBoxes);
		mvOrderedWeights.Copy(std::vector<int>(lWs.begin(), lWs.end()));

		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (mbFirstConnection && id != 0)
			{
				mpParent = tempBoxes.front();
				mpParent->AddChild(this);
				mbFirstConnection = false;
			}

		}

	}
	void ObjectBoundingBox::UpdateConnections(){
		std::map<ObjectBoundingBox*, int> BOXcounter;

		std::vector<MapPoint*> vpMPs = mvpMapPoints.get();

		//For all map points in keyframe check in which other keyframes are they seen
		//Increase counter for those keyframes
		for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
		{
			auto pMP = *vit;
			if (!pMP || pMP->isBad())
				continue;
			auto pOP = pMP->mpObjectPoint;
			if (!pOP)
				continue;
			
			std::map<ObjectBoundingBox*, size_t> observations = pOP->GetObservations();

			for (std::map<ObjectBoundingBox*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				if (mit->first->id == id)
					continue;
				BOXcounter[mit->first]++;
			}
		}
		
		// This should not happen
		if (BOXcounter.empty())
			return;

		//If the counter is greater than threshold add connection
		//In case no keyframe counter is over threshold add the one with maximum counter
		int nmax = 0;
		ObjectBoundingBox* pBOXmax = nullptr;
		int th = 10;

		std::vector<std::pair<int, ObjectBoundingBox*> > vPairs;
		//vPairs.reserve(KFcounter.size());
		for (std::map<ObjectBoundingBox*, int>::iterator mit = BOXcounter.begin(), mend = BOXcounter.end(); mit != mend; mit++)
		{
			if (mit->second > nmax)
			{
				nmax = mit->second;
				pBOXmax = mit->first;
			}
			if (mit->second >= th)
			{
				vPairs.push_back(std::make_pair(mit->second, mit->first));
				(mit->first)->AddConnection(this, mit->second);
			}
		}

		if (vPairs.empty())
		{
			vPairs.push_back(std::make_pair(nmax, pBOXmax));
			pBOXmax->AddConnection(this, nmax);
		}

		sort(vPairs.begin(), vPairs.end());
		std::list<ObjectBoundingBox*> lKFs;
		std::list<int> lWs;
		for (size_t i = 0; i < vPairs.size(); i++)
		{
			lKFs.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		mConnectedBoxWeights.Copy(BOXcounter);// = BOXcounter;
		auto tempBoxes = std::vector<ObjectBoundingBox*>(lKFs.begin(), lKFs.end());
		mvpOrderedConnectedBoxes.Copy(tempBoxes);
		mvOrderedWeights.Copy(std::vector<int>(lWs.begin(), lWs.end()));

		{
			std::unique_lock<std::mutex> lock(mMutexConnections);
			if (mbFirstConnection && id != 0)
			{
				mpParent = tempBoxes.front();
				mpParent->AddChild(this);
				mbFirstConnection = false;
			}

		}
	}
	void ObjectBoundingBox::UpdateBestCovisibles(){
	
		std::vector<std::pair<int, ObjectBoundingBox*> > vPairs;
		auto tempConnectedWeights = mConnectedBoxWeights.Get();
		for (std::map<ObjectBoundingBox*, int>::iterator mit = tempConnectedWeights.begin(), mend = tempConnectedWeights.end(); mit != mend; mit++)
			vPairs.push_back(std::make_pair(mit->second, mit->first));

		sort(vPairs.begin(), vPairs.end());
		std::list<ObjectBoundingBox*> lBoxes;
		std::list<int> lWs;
		for (size_t i = 0, iend = vPairs.size(); i < iend; i++)
		{
			lBoxes.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		mvpOrderedConnectedBoxes.Copy(std::vector<ObjectBoundingBox*>(lBoxes.begin(), lBoxes.end()));
		mvOrderedWeights.Copy(std::vector<int>(lWs.begin(), lWs.end()));
		
	}
	std::set<ObjectBoundingBox*> ObjectBoundingBox::GetConnectedBoxes(){
		return mvpOrderedConnectedBoxes.ConvertSet();
	}
	std::vector<ObjectBoundingBox* > ObjectBoundingBox::GetVectorCovisibleBoxes(){
		return mvpOrderedConnectedBoxes.get();
	}
	std::vector<ObjectBoundingBox*> ObjectBoundingBox::GetBestCovisibilityBoxes(const int& N){
		auto tempBoxes = mvpOrderedConnectedBoxes.get();
		if ((int)tempBoxes.size() < N)
			return tempBoxes;
		else
			return std::vector<ObjectBoundingBox*>(tempBoxes.begin(), tempBoxes.begin() + N);
	}
	std::vector<ObjectBoundingBox*> ObjectBoundingBox::GetCovisiblesByWeight(const int& w){
		/*if (mvpOrderedConnectedBoxes.size() == 0)
			return std::vector<ObjectBoundingBox*>();

		auto tempOrderedWeights = mvOrderedWeights.get();

		std::vector<int>::iterator it = upper_bound(tempOrderedWeights.begin(), tempOrderedWeights.end(), w, ObjectBoundingBox::weightComp);
		if (it == tempOrderedWeights.end())
			return std::vector<ObjectBoundingBox*>();
		else
		{
			int n = it - tempOrderedWeights.begin();
			return std::vector<ObjectBoundingBox*>(tempOrderedWeights.begin(), tempOrderedWeights.begin() + n);
		}*/
	}
	int ObjectBoundingBox::GetWeight(ObjectBoundingBox* pBox) {
		int res = 0;
		if (mConnectedBoxWeights.Count(pBox)) {
			res = mConnectedBoxWeights.Get(pBox);
		}
		return res;
	}
	void ObjectBoundingBox::AddChild(ObjectBoundingBox* pBox){
		if (!mspChildrens.Count(pBox))
			mspChildrens.Erase(pBox);
	}
	void ObjectBoundingBox::EraseChild(ObjectBoundingBox* pBox){
		if (mspChildrens.Count(pBox))
			mspChildrens.Erase(pBox);
	}
	void ObjectBoundingBox::ChangeParent(ObjectBoundingBox* pBox){
		{
			std::unique_lock<std::mutex> lockCon(mMutexConnections);
			mpParent = pBox;
		}
		pBox->AddChild(this);
	}
	std::set<ObjectBoundingBox*> ObjectBoundingBox::GetChilds(){
		return mspChildrens.Get();
	}
	ObjectBoundingBox* ObjectBoundingBox::GetParent(){
		std::unique_lock<std::mutex> lockCon(mMutexConnections);
		return mpParent;
	}
	bool ObjectBoundingBox::hasChild(ObjectBoundingBox* pBox){
		return mspChildrens.Count(pBox);
	}

	ObjectFrame::ObjectFrame() {}
	ObjectFrame::~ObjectFrame() {}
}