#include <Frame.h>
#include <MapPoint.h>
#include <Camera.h>
#include <CameraPose.h>
#include <FeatureDetector.h>
#include <Converter.h>

namespace EdgeSLAM {
	Frame::Frame(){}
	Frame::Frame(const Frame &frame):
		mnFrameID(frame.mnFrameID), mdTimeStamp(frame.mdTimeStamp), mpCamPose(frame.mpCamPose), mpCamera(frame.mpCamera),
		K(frame.K), D(frame.D),fx(frame.fx), fy(frame.fy), cx(frame.cx), cy(frame.cy), invfx(frame.invfx), invfy(frame.invfy),
		mnMinX(frame.mnMinX), mnMaxX(frame.mnMaxX), mnMinY(frame.mnMinY), mnMaxY(frame.mnMaxY), mfGridElementWidthInv(frame.mfGridElementWidthInv), mfGridElementHeightInv(frame.mfGridElementHeightInv),
		FRAME_GRID_COLS(frame.FRAME_GRID_COLS), FRAME_GRID_ROWS(frame.FRAME_GRID_ROWS), mbDistorted(frame.mbDistorted),
		mnScaleLevels(frame.mnScaleLevels), mfScaleFactor(frame.mfScaleFactor),
		mfLogScaleFactor(frame.mfLogScaleFactor),
		mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
		mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
	{
		
	}
	Frame::Frame(cv::Mat img, Camera* pCam, int id, double time_stamp):mnFrameID(id), mdTimeStamp(time_stamp), mpCamera(pCam),
		K(pCam->K), D(pCam->D),fx(pCam->fx),fy(pCam->fy), cx(pCam->cx), cy(pCam->cy), invfx(pCam->invfx), invfy(pCam->invfy), mnMinX(pCam->u_min), mnMaxX(pCam->u_max), mnMinY(pCam->v_min), mnMaxY(pCam->v_max), mfGridElementWidthInv(pCam->mfGridElementWidthInv), mfGridElementHeightInv(pCam->mfGridElementHeightInv), FRAME_GRID_COLS(pCam->mnGridCols), FRAME_GRID_ROWS(pCam->mnGridRows), mbDistorted(pCam->bDistorted),
		mnScaleLevels(detector->mnScaleLevels), mfScaleFactor(detector->mfScaleFactor), mfLogScaleFactor(detector->mfLogScaleFactor), mvScaleFactors(detector->mvScaleFactors), mvInvScaleFactors(detector->mvInvScaleFactors), mvLevelSigma2(detector->mvLevelSigma2), mvInvLevelSigma2(detector->mvInvLevelSigma2)
	{
		mpCamPose = new CameraPose();
		imgColor = img.clone();
		cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);//COLOR_BGR2GRAY
		detector->detectAndCompute(imgGray, cv::Mat(), mvKeys, mDescriptors);
		N = mvKeys.size();

		if (mbDistorted)
			UndistortKeyPoints();
		else
			mvKeysUn = mvKeys;

		mGrid = new std::vector<size_t>*[FRAME_GRID_COLS];
		for (int i = 0; i < FRAME_GRID_COLS; i++)
			mGrid[i] = new std::vector<size_t>[FRAME_GRID_ROWS];
		
		AssignFeaturesToGrid();
	}
	Frame::~Frame(){}

	bool Frame::is_in_frustum(MapPoint* pMP, float viewingCosLimit) {
		pMP->mbTrackInView = false;

		// 3D in absolute coordinates
		cv::Mat P = pMP->GetWorldPos();

		cv::Mat Rw = mpCamPose->GetRotation();
		cv::Mat tw = mpCamPose->GetTranslation();
		cv::Mat Ow = mpCamPose->GetCenter();

		// 3D in camera coordinates
		const cv::Mat Pc = Rw*P + tw;
		const float &PcX = Pc.at<float>(0);
		const float &PcY = Pc.at<float>(1);
		const float &PcZ = Pc.at<float>(2);

		// Check positive depth
		if (PcZ<0.0f)
			return false;

		// Project in image and check it is not outside
		const float invz = 1.0f / PcZ;
		const float u = fx*PcX*invz + cx;
		const float v = fy*PcY*invz + cy;
		
		if (u<mnMinX || u>mnMaxX || v < mnMinY || v > mnMaxY)
			return false;

		// Check distance is in the scale invariance region of the MapPoint
		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		const cv::Mat PO = P - Ow;
		const float dist = cv::norm(PO);

		if (dist<minDistance || dist>maxDistance)
			return false;

		// Check viewing angle
		cv::Mat Pn = pMP->GetNormal();
		const float viewCos = PO.dot(Pn) / dist;

		if (viewCos<viewingCosLimit)
			return false;

		// Predict scale in the image
		const int nPredictedLevel = pMP->PredictScale(dist, this);

		// Data used by the tracking
		pMP->mbTrackInView = true;
		pMP->mTrackProjX = u;
		pMP->mTrackProjY = v;
		pMP->mnTrackScaleLevel = nPredictedLevel;
		pMP->mTrackViewCos = viewCos;

		return true;
	}

	bool Frame::is_in_image(float x, float y, float z){
		return mpCamera->is_in_image(x, y, z);
	}
	void Frame::reset_map_points(){
		mvpMapPoints = std::vector<MapPoint*>(mvKeysUn.size(), nullptr);
		mvbOutliers = std::vector<bool>(mvKeysUn.size(), false);
	}

	void Frame::check_replaced_map_points() {
		for (int i = 0; i < N; i++) {
			auto pMP = mvpMapPoints[i];
			if (pMP) {
				auto pMPrep = pMP->GetReplaced();
				if (pMPrep)
					mvpMapPoints[i] = pMPrep;
			}
		}
	}
	
	void Frame::ComputeBoW()
	{
		if (mBowVec.empty())
		{
			std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
			mpVoc->transform(vCurrentDesc, mBowVec, mFeatVec, 4);  // 5 is better
		}
	}

	void Frame::SetPose(const cv::Mat &Tcw) {
		mpCamPose->SetPose(Tcw);
	}
	cv::Mat Frame::GetPose(){
		return mpCamPose->GetPose();
	}
	cv::Mat Frame::GetPoseInverse(){
		return mpCamPose->GetInversePose();
	}
	cv::Mat Frame::GetCameraCenter(){
		return mpCamPose->GetCenter();
	}
	cv::Mat Frame::GetRotation(){
		return mpCamPose->GetRotation();
	}
	cv::Mat Frame::GetTranslation(){
		return mpCamPose->GetTranslation();
	}

	std::vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel)const{
		std::vector<size_t> vIndices;
		vIndices.reserve(N);

		const int nMinCellX = std::max(0, (int)floor((x - mnMinX - r)*mfGridElementWidthInv));
		if (nMinCellX >= FRAME_GRID_COLS)
			return vIndices;

		const int nMaxCellX = std::min((int)FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r)*mfGridElementWidthInv));
		if (nMaxCellX<0)
			return vIndices;

		const int nMinCellY = std::max(0, (int)floor((y - mnMinY - r)*mfGridElementHeightInv));
		if (nMinCellY >= FRAME_GRID_ROWS)
			return vIndices;

		const int nMaxCellY = std::min((int)FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r)*mfGridElementHeightInv));
		if (nMaxCellY<0)
			return vIndices;

		const bool bCheckLevels = (minLevel>0) || (maxLevel >= 0);
		bool bCheckMinLevel = minLevel > 0;
		bool bCheckMaxLevel = maxLevel >= 0;

		for (int ix = nMinCellX; ix <= nMaxCellX; ix++)
		{
			for (int iy = nMinCellY; iy <= nMaxCellY; iy++)
			{
				const std::vector<size_t> vCell = mGrid[ix][iy];
				if (vCell.empty())
					continue;

				for (size_t j = 0, jend = vCell.size(); j<jend; j++)
				{
					const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
					if(bCheckMinLevel && kpUn.octave<minLevel)
						continue;
					if (bCheckMaxLevel && kpUn.octave > maxLevel)
						continue;

					const float distx = kpUn.pt.x - x;
					const float disty = kpUn.pt.y - y;

					if (fabs(distx)<r && fabs(disty)<r)
						vIndices.push_back(vCell[j]);
				}
			}
		}

		return vIndices;
	}
	void Frame::UndistortKeyPoints(){
		cv::Mat mat(N, 2, CV_32F);
		for (int i = 0; i<N; i++)
		{
			mat.at<float>(i, 0) = mvKeys[i].pt.x;
			mat.at<float>(i, 1) = mvKeys[i].pt.y;
		}

		// Undistort points
		mat = mat.reshape(2);
		cv::undistortPoints(mat, mat, K, D, cv::Mat(), K);
		mat = mat.reshape(1);

		// Fill undistorted keypoint vector
		mvKeysUn.resize(N);
		for (int i = 0; i<N; i++)
		{
			cv::KeyPoint kp = mvKeys[i];
			kp.pt.x = mat.at<float>(i, 0);
			kp.pt.y = mat.at<float>(i, 1);
			mvKeysUn[i] = kp;
		}
	}
	void Frame::AssignFeaturesToGrid(){
		int nReserve = 0.5f*N / (FRAME_GRID_COLS*FRAME_GRID_ROWS);
		for (unsigned int i = 0; i<FRAME_GRID_COLS; i++)
			for (unsigned int j = 0; j<FRAME_GRID_ROWS; j++)
				mGrid[i][j].reserve(nReserve);

		for (int i = 0; i<N; i++)
		{
			const cv::KeyPoint &kp = mvKeysUn[i];

			int nGridPosX, nGridPosY;
			if (PosInGrid(kp, nGridPosX, nGridPosY))
				mGrid[nGridPosX][nGridPosY].push_back(i);
		}
	}
	bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY){
		posX = round((kp.pt.x - mnMinX)*mfGridElementWidthInv);
		posY = round((kp.pt.y - mnMinY)*mfGridElementHeightInv);

		if (posX<0 || posX >= FRAME_GRID_COLS || posY<0 || posY >= FRAME_GRID_ROWS)
			return false;

		return true;
	}

	void Frame::TurnOnFlag(unsigned char opt) {
		std::unique_lock<std::mutex>(mMutexFlag);
		mnFlag |= opt;
	}
	void Frame::TurnOffFlag(unsigned char opt) {
		std::unique_lock<std::mutex>(mMutexFlag);
		mnFlag &= ~opt;
	}
	bool Frame::CheckFlag(unsigned char opt) {
		std::unique_lock<std::mutex>(mMutexFlag);
		unsigned char flag = mnFlag & opt;
		return flag == opt;
	}
}