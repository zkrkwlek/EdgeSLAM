#include <Optimizer.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <Map.h>
#include <MapPoint.h>
#include <Converter.h>

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/linear_solver_eigen.h"
#include "g2o/types/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/types/types_seven_dof_expmap.h"

namespace EdgeSLAM {
	void Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
	{
		std::vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
		std::vector<MapPoint*> vpMP = pMap->GetAllMapPoints();
		BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
	}


	void Optimizer::BundleAdjustment(const std::vector<KeyFrame *> &vpKFs, const std::vector<MapPoint *> &vpMP,
		int nIterations, bool* pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
	{
		std::vector<bool> vbNotIncludedMP;
		vbNotIncludedMP.resize(vpMP.size());

		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

		linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		if (pbStopFlag)
			optimizer.setForceStopFlag(pbStopFlag);

		long unsigned int maxKFid = 0;

		// Set KeyFrame vertices
		for (size_t i = 0; i<vpKFs.size(); i++)
		{
			KeyFrame* pKF = vpKFs[i];
			if (pKF->isBad())
				continue;
			g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
			vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
			vSE3->setId(pKF->mnId);
			vSE3->setFixed(pKF->mnId == 0);
			optimizer.addVertex(vSE3);
			if (pKF->mnId>maxKFid)
				maxKFid = pKF->mnId;
		}

		const float thHuber2D = sqrt(5.99);

		// Set MapPoint vertices
		for (size_t i = 0; i<vpMP.size(); i++)
		{
			MapPoint* pMP = vpMP[i];
			if (pMP->isBad())
				continue;
			g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
			vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
			const int id = pMP->mnId + maxKFid + 1;
			vPoint->setId(id);
			vPoint->setMarginalized(true);
			optimizer.addVertex(vPoint);

			const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

			int nEdges = 0;
			//SET EDGES
			for (std::map<KeyFrame*, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
			{

				KeyFrame* pKF = mit->first;
				if (pKF->isBad() || pKF->mnId>maxKFid)
					continue;

				nEdges++;

				const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

				Eigen::Matrix<double, 2, 1> obs;
				obs << kpUn.pt.x, kpUn.pt.y;

				g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
				e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKF->mnId)));
				e->setMeasurement(obs);
				const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
				e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

				if (bRobust)
				{
					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuber2D);
				}

				e->fx = pKF->fx;
				e->fy = pKF->fy;
				e->cx = pKF->cx;
				e->cy = pKF->cy;

				optimizer.addEdge(e);
			}

			if (nEdges == 0)
			{
				optimizer.removeVertex(vPoint);
				vbNotIncludedMP[i] = true;
			}
			else
			{
				vbNotIncludedMP[i] = false;
			}
		}

		// Optimize!
		optimizer.initializeOptimization();
		optimizer.optimize(nIterations);

		// Recover optimized data

		//Keyframes
		for (size_t i = 0; i<vpKFs.size(); i++)
		{
			KeyFrame* pKF = vpKFs[i];
			if (pKF->isBad())
				continue;
			g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
			g2o::SE3Quat SE3quat = vSE3->estimate();
			if (nLoopKF == 0)
			{
				pKF->SetPose(Converter::toCvMat(SE3quat));
			}
			else
			{
				pKF->mTcwGBA.create(4, 4, CV_32F);
				Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
				pKF->mnBAGlobalForKF = nLoopKF;
			}
		}

		//Points
		for (size_t i = 0; i<vpMP.size(); i++)
		{
			if (vbNotIncludedMP[i])
				continue;

			MapPoint* pMP = vpMP[i];

			if (pMP->isBad())
				continue;
			g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));

			if (nLoopKF == 0)
			{
				pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
				pMP->UpdateNormalAndDepth();
			}
			else
			{
				pMP->mPosGBA.create(3, 1, CV_32F);
				Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
				pMP->mnBAGlobalForKF = nLoopKF;
			}
		}

	}

	int Optimizer::PoseOptimization(Frame *pFrame)
	{
		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

		linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		int nInitialCorrespondences = 0;

		// Set Frame vertex
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setEstimate(Converter::toSE3Quat(pFrame->GetPose()));
		vSE3->setId(0);
		vSE3->setFixed(false);
		optimizer.addVertex(vSE3);

		// Set MapPoint vertices
		const int N = pFrame->N;

		std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
		std::vector<size_t> vnIndexEdgeMono;
		vpEdgesMono.reserve(N);
		vnIndexEdgeMono.reserve(N);
		
		const float deltaMono = sqrt(5.991);


		{
			//std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

			for (int i = 0; i<N; i++)
			{
				MapPoint* pMP = pFrame->mvpMapPoints[i];
				if (pMP)
				{
					nInitialCorrespondences++;
					pFrame->mvbOutliers[i] = false;

					Eigen::Matrix<double, 2, 1> obs;
					const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
					obs << kpUn.pt.x, kpUn.pt.y;

					g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
					e->setMeasurement(obs);
					const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(deltaMono);

					e->fx = pFrame->fx;
					e->fy = pFrame->fy;
					e->cx = pFrame->cx;
					e->cy = pFrame->cy;
					cv::Mat Xw = pMP->GetWorldPos();
					e->Xw[0] = Xw.at<float>(0);
					e->Xw[1] = Xw.at<float>(1);
					e->Xw[2] = Xw.at<float>(2);

					optimizer.addEdge(e);

					vpEdgesMono.push_back(e);
					vnIndexEdgeMono.push_back(i);
				}

			}
		}


		if (nInitialCorrespondences<3)
			return 0;

		// We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
		// At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
		const float chi2Mono[4] = { 5.991,5.991,5.991,5.991 };
		const int its[4] = { 10,10,10,10 };

		int nBad = 0;
		for (size_t it = 0; it<4; it++)
		{

			vSE3->setEstimate(Converter::toSE3Quat(pFrame->GetPose()));
			optimizer.initializeOptimization(0);
			optimizer.optimize(its[it]);

			nBad = 0;
			for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
			{
				g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

				const size_t idx = vnIndexEdgeMono[i];

				if (pFrame->mvbOutliers[idx])
				{
					e->computeError();
				}

				const float chi2 = e->chi2();

				if (chi2>chi2Mono[it])
				{
					pFrame->mvbOutliers[idx] = true;
					e->setLevel(1);
					nBad++;
				}
				else
				{
					pFrame->mvbOutliers[idx] = false;
					e->setLevel(0);
				}

				if (it == 2)
					e->setRobustKernel(0);
			}

			if (optimizer.edges().size()<10)
				break;
		}

		// Recover optimized pose and return number of inliers
		g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
		g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
		cv::Mat pose = Converter::toCvMat(SE3quat_recov);
		pFrame->SetPose(pose);

		return nInitialCorrespondences - nBad;
	}

	void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
	{
		// Local KeyFrames: First Breath Search from Current Keyframe
		std::list<KeyFrame*> lLocalKeyFrames;

		lLocalKeyFrames.push_back(pKF);
		pKF->mnBALocalForKF = pKF->mnId;

		const std::vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
		for (int i = 0, iend = vNeighKFs.size(); i<iend; i++)
		{
			KeyFrame* pKFi = vNeighKFs[i];
			pKFi->mnBALocalForKF = pKF->mnId;
			if (!pKFi->isBad())
				lLocalKeyFrames.push_back(pKFi);
		}

		// Local MapPoints seen in Local KeyFrames
		std::list<MapPoint*> lLocalMapPoints;
		for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			std::vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
			for (std::vector<MapPoint*>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
			{
				MapPoint* pMP = *vit;
				if (pMP)
					if (!pMP->isBad())
						if (pMP->mnBALocalForKF != pKF->mnId)
						{
							lLocalMapPoints.push_back(pMP);
							pMP->mnBALocalForKF = pKF->mnId;
						}
			}
		}

		// Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
		std::list<KeyFrame*> lFixedCameras;
		for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
		{
			std::map<KeyFrame*, size_t> observations = (*lit)->GetObservations();
			for (std::map<KeyFrame*, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				KeyFrame* pKFi = mit->first;

				if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
				{
					pKFi->mnBAFixedForKF = pKF->mnId;
					if (!pKFi->isBad())
						lFixedCameras.push_back(pKFi);
				}
			}
		}

		// Setup optimizer
		g2o::SparseOptimizer optimizer;
		g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

		linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

		g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

		g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		optimizer.setAlgorithm(solver);

		if (pbStopFlag)
			optimizer.setForceStopFlag(pbStopFlag);

		unsigned long maxKFid = 0;

		// Set Local KeyFrame vertices
		for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			KeyFrame* pKFi = *lit;
			g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
			vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
			vSE3->setId(pKFi->mnId);
			vSE3->setFixed(pKFi->mnId == 0);
			optimizer.addVertex(vSE3);
			if (pKFi->mnId>maxKFid)
				maxKFid = pKFi->mnId;
		}

		// Set Fixed KeyFrame vertices
		for (std::list<KeyFrame*>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
		{
			KeyFrame* pKFi = *lit;
			g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
			vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
			vSE3->setId(pKFi->mnId);
			vSE3->setFixed(true);
			optimizer.addVertex(vSE3);
			if (pKFi->mnId>maxKFid)
				maxKFid = pKFi->mnId;
		}

		// Set MapPoint vertices
		const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size())*lLocalMapPoints.size();

		std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
		vpEdgesMono.reserve(nExpectedSize);

		std::vector<KeyFrame*> vpEdgeKFMono;
		vpEdgeKFMono.reserve(nExpectedSize);

		std::vector<MapPoint*> vpMapPointEdgeMono;
		vpMapPointEdgeMono.reserve(nExpectedSize);

		const float thHuberMono = sqrt(5.991);

		for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
		{
			MapPoint* pMP = *lit;
			g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
			vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
			int id = pMP->mnId + maxKFid + 1;
			vPoint->setId(id);
			vPoint->setMarginalized(true);
			optimizer.addVertex(vPoint);

			const std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

			//Set edges
			for (std::map<KeyFrame*, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				KeyFrame* pKFi = mit->first;

				if (!pKFi->isBad())
				{
					const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

					Eigen::Matrix<double, 2, 1> obs;
					obs << kpUn.pt.x, kpUn.pt.y;

					g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

					e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
					e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
					e->setMeasurement(obs);
					const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
					e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(thHuberMono);

					e->fx = pKFi->fx;
					e->fy = pKFi->fy;
					e->cx = pKFi->cx;
					e->cy = pKFi->cy;

					optimizer.addEdge(e);
					vpEdgesMono.push_back(e);
					vpEdgeKFMono.push_back(pKFi);
					vpMapPointEdgeMono.push_back(pMP);
				}
			}
		}

		if (pbStopFlag)
			if (*pbStopFlag)
				return;

		optimizer.initializeOptimization();
		optimizer.optimize(5);

		bool bDoMore = true;

		if (pbStopFlag)
			if (*pbStopFlag)
				bDoMore = false;

		if (bDoMore)
		{

			// Check inlier observations
			for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
			{
				g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
				MapPoint* pMP = vpMapPointEdgeMono[i];

				if (pMP->isBad())
					continue;

				if (e->chi2()>5.991 || !e->isDepthPositive())
				{
					e->setLevel(1);
				}

				e->setRobustKernel(0);
			}
			// Optimize again without the outliers
			optimizer.initializeOptimization(0);
			optimizer.optimize(10);

		}

		std::vector<std::pair<KeyFrame*, MapPoint*> > vToErase;
		vToErase.reserve(vpEdgesMono.size());

		// Check inlier observations       
		for (size_t i = 0, iend = vpEdgesMono.size(); i<iend; i++)
		{
			g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
			MapPoint* pMP = vpMapPointEdgeMono[i];

			if (pMP->isBad())
				continue;

			if (e->chi2()>5.991 || !e->isDepthPositive())
			{
				KeyFrame* pKFi = vpEdgeKFMono[i];
				vToErase.push_back(std::make_pair(pKFi, pMP));
			}
		}

		// Get Map Mutex
		std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

		if (!vToErase.empty())
		{
			for (size_t i = 0; i<vToErase.size(); i++)
			{
				KeyFrame* pKFi = vToErase[i].first;
				MapPoint* pMPi = vToErase[i].second;
				pKFi->EraseMapPointMatch(pMPi);
				pMPi->EraseObservation(pKFi);
			}
		}

		// Recover optimized data

		//Keyframes
		for (std::list<KeyFrame*>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
		{
			KeyFrame* pKF = *lit;
			g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
			g2o::SE3Quat SE3quat = vSE3->estimate();
			pKF->SetPose(Converter::toCvMat(SE3quat));
		}

		//Points
		for (std::list<MapPoint*>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
		{
			MapPoint* pMP = *lit;
			g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId + maxKFid + 1));
			pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
			pMP->UpdateNormalAndDepth();
		}
	}
}