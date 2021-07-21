#include <PnPSolver.h>
#include <random>
#include <Frame.h>
#include <MapPoint.h>

namespace EdgeSLAM {
	PnPSolver::PnPSolver(Frame *F, std::vector<MapPoint*> vpMapPointMatches):
		pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
		mnIterations(0), mnBestInliers(0), N(0)
	{
		mvpMapPointMatches = vpMapPointMatches;
		mvP2D.reserve(F->mvpMapPoints.size());
		mvSigma2.reserve(F->mvpMapPoints.size());
		mvP3Dw.reserve(F->mvpMapPoints.size());
		mvKeyPointIndices.reserve(F->mvpMapPoints.size());
		mvAllIndices.reserve(F->mvpMapPoints.size());

		int idx = 0;
		for (size_t i = 0, iend = vpMapPointMatches.size(); i<iend; i++)
		{
			MapPoint* pMP = vpMapPointMatches[i];

			if (pMP)
			{
				if (!pMP->isBad())
				{
					const cv::KeyPoint &kp = F->mvKeysUn[i];

					mvP2D.push_back(kp.pt);
					mvSigma2.push_back(F->mvLevelSigma2[kp.octave]);

					cv::Mat Pos = pMP->GetWorldPos();
					mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0), Pos.at<float>(1), Pos.at<float>(2)));

					mvKeyPointIndices.push_back(i);
					mvAllIndices.push_back(idx);

					idx++;
				}
			}
		}

		cws = cv::Mat::zeros(4, 3, CV_64FC1);
		ccs = cv::Mat::zeros(4, 3, CV_64FC1);

		// Set camera calibration parameters
		fu = F->fx;
		fv = F->fy;
		uc = F->cx;
		vc = F->cy;

		SetRansacParameters();
	}
	PnPSolver::~PnPSolver(){
		delete[] pws;
		delete[] us;
		delete[] alphas;
		delete[] pcs;
	}

	void PnPSolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
	{
		mRansacProb = probability;
		mRansacMinInliers = minInliers;
		mRansacMaxIts = maxIterations;
		mRansacEpsilon = epsilon;
		mRansacMinSet = minSet;

		N = mvP2D.size(); // number of correspondences

		mvbInliersi.resize(N);

		// Adjust Parameters according to number of correspondences
		int nMinInliers = N*mRansacEpsilon;
		if (nMinInliers<mRansacMinInliers)
			nMinInliers = mRansacMinInliers;
		if (nMinInliers<minSet)
			nMinInliers = minSet;
		mRansacMinInliers = nMinInliers;

		if (mRansacEpsilon<(float)mRansacMinInliers / N)
			mRansacEpsilon = (float)mRansacMinInliers / N;

		// Set RANSAC iterations according to probability, epsilon, and max iterations
		int nIterations;

		if (mRansacMinInliers == N)
			nIterations = 1;
		else
			nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(mRansacEpsilon, 3)));

		mRansacMaxIts = std::max(1, std::min(nIterations, mRansacMaxIts));

		mvMaxError.resize(mvSigma2.size());
		for (size_t i = 0; i<mvSigma2.size(); i++)
			mvMaxError[i] = mvSigma2[i] * th2;
	}

	cv::Mat PnPSolver::find(std::vector<bool> &vbInliers, int &nInliers)
	{
		bool bFlag;
		return iterate(mRansacMaxIts, bFlag, vbInliers, nInliers);
	}

	cv::Mat PnPSolver::iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers)
	{
		bNoMore = false;
		vbInliers.clear();
		nInliers = 0;

		set_maximum_number_of_correspondences(mRansacMinSet);

		if (N<mRansacMinInliers)
		{
			bNoMore = true;
			return cv::Mat();
		}

		std::vector<size_t> vAvailableIndices;


		std::random_device rn;
		std::mt19937_64 rnd(rn());

		int nCurrentIterations = 0;
		while (mnIterations<mRansacMaxIts || nCurrentIterations<nIterations)
		{
			nCurrentIterations++;
			mnIterations++;
			reset_correspondences();
			vAvailableIndices = mvAllIndices;
			std::uniform_int_distribution<int> range(0, vAvailableIndices.size() - 1);

			// Get min set of points
			for (short i = 0; i < mRansacMinSet; ++i)
			{
				int randi = range(rnd);
				int idx = vAvailableIndices[randi];

				add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z, mvP2D[idx].x, mvP2D[idx].y);

				vAvailableIndices[randi] = vAvailableIndices.back();
				vAvailableIndices.pop_back();
			}

			// Compute camera pose
			compute_pose(mRi, mti);

			// Check inliers
			CheckInliers();

			if (mnInliersi >= mRansacMinInliers)
			{
				// If it is the best solution so far, save it
				if (mnInliersi>mnBestInliers)
				{
					mvbBestInliers = mvbInliersi;
					mnBestInliers = mnInliersi;

					cv::Mat Rcw = mRi.clone();
					cv::Mat tcw = mti.clone();
					Rcw.convertTo(Rcw, CV_32F);
					tcw.convertTo(tcw, CV_32F);
					mBestTcw = cv::Mat::eye(4, 4, CV_32F);
					Rcw.copyTo(mBestTcw.rowRange(0, 3).colRange(0, 3));
					tcw.copyTo(mBestTcw.rowRange(0, 3).col(3));
				}

				if (Refine())
				{
					nInliers = mnRefinedInliers;
					vbInliers = std::vector<bool>(mvpMapPointMatches.size(), false);
					for (int i = 0; i<N; i++)
					{
						if (mvbRefinedInliers[i])
							vbInliers[mvKeyPointIndices[i]] = true;
					}
					return mRefinedTcw.clone();
				}

			}
		}

		if (mnIterations >= mRansacMaxIts)
		{
			bNoMore = true;
			if (mnBestInliers >= mRansacMinInliers)
			{
				nInliers = mnBestInliers;
				vbInliers = std::vector<bool>(mvpMapPointMatches.size(), false);
				for (int i = 0; i<N; i++)
				{
					if (mvbBestInliers[i])
						vbInliers[mvKeyPointIndices[i]] = true;
				}
				return mBestTcw.clone();
			}
		}

		return cv::Mat();
	}

	bool PnPSolver::Refine()
	{
		std::vector<int> vIndices;
		vIndices.reserve(mvbBestInliers.size());

		for (size_t i = 0; i<mvbBestInliers.size(); i++)
		{
			if (mvbBestInliers[i])
			{
				vIndices.push_back(i);
			}
		}

		set_maximum_number_of_correspondences(vIndices.size());

		reset_correspondences();

		for (size_t i = 0; i<vIndices.size(); i++)
		{
			int idx = vIndices[i];
			add_correspondence(mvP3Dw[idx].x, mvP3Dw[idx].y, mvP3Dw[idx].z, mvP2D[idx].x, mvP2D[idx].y);
		}

		// Compute camera pose
		compute_pose(mRi, mti);

		// Check inliers
		CheckInliers();

		mnRefinedInliers = mnInliersi;
		mvbRefinedInliers = mvbInliersi;

		if (mnInliersi>mRansacMinInliers)
		{
			cv::Mat Rcw, tcw;
			mRi.convertTo(Rcw, CV_32F);
			mti.convertTo(tcw, CV_32F);
			mRefinedTcw = cv::Mat::eye(4, 4, CV_32F);
			Rcw.copyTo(mRefinedTcw.rowRange(0, 3).colRange(0, 3));
			tcw.copyTo(mRefinedTcw.rowRange(0, 3).col(3));
			return true;
		}

		return false;
	}


	void PnPSolver::CheckInliers()
	{
		mnInliersi = 0;

		for (int i = 0; i<N; i++)
		{
			cv::Point3f P3Dw = mvP3Dw[i];
			cv::Point2f P2D = mvP2D[i];

			cv::Mat X = mRi*cv::Mat(P3Dw);

			float Xc = X.at<float>(0);
			float Yc = X.at<float>(1);
			float invZc = 1 / X.at<float>(2);;

			double ue = uc + fu * Xc * invZc;
			double ve = vc + fv * Yc * invZc;

			float distX = P2D.x - ue;
			float distY = P2D.y - ve;

			float error2 = distX*distX + distY*distY;

			if (error2<mvMaxError[i])
			{
				mvbInliersi[i] = true;
				mnInliersi++;
			}
			else
			{
				mvbInliersi[i] = false;
			}
		}
	}


	void PnPSolver::set_maximum_number_of_correspondences(int n)
	{
		if (maximum_number_of_correspondences < n) {
			if (pws != 0) delete[] pws;
			if (us != 0) delete[] us;
			if (alphas != 0) delete[] alphas;
			if (pcs != 0) delete[] pcs;

			maximum_number_of_correspondences = n;
			pws = new double[3 * maximum_number_of_correspondences];
			us = new double[2 * maximum_number_of_correspondences];
			alphas = new double[4 * maximum_number_of_correspondences];
			pcs = new double[3 * maximum_number_of_correspondences];
		}
	}

	void PnPSolver::reset_correspondences(void)
	{
		number_of_correspondences = 0;
	}

	void PnPSolver::add_correspondence(double X, double Y, double Z, double u, double v)
	{
		pws[3 * number_of_correspondences] = X;
		pws[3 * number_of_correspondences + 1] = Y;
		pws[3 * number_of_correspondences + 2] = Z;

		us[2 * number_of_correspondences] = u;
		us[2 * number_of_correspondences + 1] = v;

		number_of_correspondences++;
	}

	void PnPSolver::choose_control_points(void)
	{
		// Take C0 as the reference points centroid:
		cws.at<double>(0, 0) = 0.0;
		cws.at<double>(0, 1) = 0.0;
		cws.at<double>(0, 2) = 0.0;
		for (int i = 0; i < number_of_correspondences; i++)
			for (int j = 0; j < 3; j++)
				cws.at<double>(0,j) += pws[3 * i + j];

		for (int j = 0; j < 3; j++)
			cws.at<double>(0, j) /= number_of_correspondences;


		// Take C1, C2, and C3 from PCA on the reference points:
		cv::Mat PWO = cv::Mat::zeros(number_of_correspondences, 3, CV_64FC1);
		cv::Mat PW0tPW0 = cv::Mat::zeros(3, 3, CV_64FC1);
		cv::Mat DC		= cv::Mat::zeros(3, 1, CV_64FC1);
		cv::Mat UCt		= cv::Mat::zeros(3, 3, CV_64FC1);
		
		for (int i = 0; i < number_of_correspondences; i++)
			for (int j = 0; j < 3; j++)
			{
				PWO.at<double>(i, j) = pws[3 * i + j] - cws.at<double>(0, j);
			}

		cv::mulTransposed(PWO, PW0tPW0, true);

		cv::SVD::compute(PW0tPW0, DC, UCt, cv::Mat(), cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

		for (int i = 1; i < 4; i++) {
			double k = sqrt(DC.at<double>(i-1) / number_of_correspondences);
			for (int j = 0; j < 3; j++){
				cws.at<double>(i, j) = cws.at<double>(0, j) + k*UCt.at<double>(i - 1, j);
			}
		}
	}

	void PnPSolver::compute_barycentric_coordinates(void)
	{
		cv::Mat CC = cv::Mat::zeros(3, 3, CV_64F);
		
		for (int i = 0; i < 3; i++)
			for (int j = 1; j < 4; j++)
				CC.at<double>(i, j) = cws.at<double>(j, i) - cws.at<double>(0, i);
		cv::Mat CC_inv = CC.inv(cv::DECOMP_SVD);
		for (int i = 0; i < number_of_correspondences; i++) {
			double * pi = pws + 3 * i;
			double * a = alphas + 4 * i;

			for (int j = 0; j < 3; j++)
				a[1 + j] =
				CC_inv.at<double>(j, 0) * (pi[0] - cws.at<double>(0, 0)) +
				CC_inv.at<double>(j, 1) * (pi[1] - cws.at<double>(0, 1)) +
				CC_inv.at<double>(j, 2) * (pi[2] - cws.at<double>(0, 2));
			a[0] = 1.0f - a[1] - a[2] - a[3];
		}
	}

	void PnPSolver::fill_M(cv::Mat& M, const int row, const double * as, const double u, const double v)
	{
		int row2 = row + 1;
		for (int i = 0; i < 4; i++) {
			M.at<double>(row, i) = as[i] * fu;
			M.at<double>(row, i + 1) = 0.0;
			M.at<double>(row, i+2) = as[i] * (uc - u);

			M.at<double>(row2, i) = 0.0;
			M.at<double>(row2, i+1) = as[i] * fv;
			M.at<double>(row2, i+2) = as[i] * (vc - v);
		}
	}

	void PnPSolver::compute_ccs(cv::Mat betas, cv::Mat ut)
	{
		for (int i = 0; i < 4; i++)
			ccs.at<double>(i, 0) = ccs.at<double>(i, 1) = ccs.at<double>(i, 2) = 0.0;

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++)
				for (int k = 0; k < 3; k++)
					ccs.at<double>(j, k) += betas.at<double>(i) * ut.at<double>(11 - i, 3 * j + k);
		}
	}

	void PnPSolver::compute_pcs(void)
	{
		for (int i = 0; i < number_of_correspondences; i++) {
			double * a = alphas + 4 * i;
			double * pc = pcs + 3 * i;

			for (int j = 0; j < 3; j++)
				pc[j] = a[0] * ccs.at<double>(0,j) + a[1] * ccs.at<double>(1, j) + a[2] * ccs.at<double>(2, j) + a[3] * ccs.at<double>(3, j);
		}
	}

	double PnPSolver::compute_pose(cv::Mat R, cv::Mat t)
	{
		choose_control_points();
		compute_barycentric_coordinates();

		cv::Mat M = cv::Mat::zeros(2 * number_of_correspondences, 12, CV_64F);

		for (int i = 0; i < number_of_correspondences; i++)
			fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);

		cv::Mat MtM = cv::Mat::zeros(12, 12, CV_64FC1);
		cv::Mat D = cv::Mat::zeros(12, 1, CV_64FC1);
		cv::Mat Ut = cv::Mat::zeros(12, 12, CV_64FC1);
		cv::mulTransposed(M, MtM, true);
		cv::SVD::compute(MtM, D, Ut, cv::Mat(), cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
		
		cv::Mat L_6x10 = cv::Mat::zeros(6, 10, CV_64FC1);
		cv::Mat Rho = cv::Mat::zeros(6, 1, CV_64FC1);

		compute_L_6x10(Ut, L_6x10);
		compute_rho(Rho);

		std::vector<cv::Mat> Betas(4, cv::Mat::zeros(4, 1, CV_64FC1));
		std::vector<cv::Mat> rep_errors(4, cv::Mat::zeros(1, 1, CV_64FC1));
		std::vector<cv::Mat> Rs(4, cv::Mat::zeros(3,3,CV_64FC1));
		std::vector<cv::Mat> ts(4, cv::Mat::zeros(3, 1, CV_64FC1));

		/*double Betas[4][4], rep_errors[4];
		double Rs[4][3][3], ts[4][3];*/

		find_betas_approx_1(L_6x10, Rho, Betas[1]);
		gauss_newton(L_6x10, Rho, Betas[1]);
		rep_errors[1] = compute_R_and_t(Ut, Betas[1], Rs[1], ts[1]);

		find_betas_approx_2(L_6x10, Rho, Betas[2]);
		gauss_newton(L_6x10, Rho, Betas[2]);
		rep_errors[2] = compute_R_and_t(Ut, Betas[2], Rs[2], ts[2]);

		find_betas_approx_3(L_6x10, Rho, Betas[3]);
		gauss_newton(L_6x10, Rho, Betas[3]);
		rep_errors[3] = compute_R_and_t(Ut, Betas[3], Rs[3], ts[3]);

		int N = 1;
		if (rep_errors[2].at<double>(0) < rep_errors[1].at<double>(0)) N = 2;
		if (rep_errors[3].at<double>(0) < rep_errors[N].at<double>(0)) N = 3;

		Rs[N].copyTo(R);
		ts[N].copyTo(t);

		return rep_errors[N].at<double>(0);
	}
	double PnPSolver::dot(const double * v1, const double * v2) {
		return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	}
	double PnPSolver::dist2(cv::Mat p1, cv::Mat p2)
	{
		cv::Mat res = p1 - p2;
		return res.dot(res);
	}
	
	double PnPSolver::reprojection_error(cv::Mat R, cv::Mat t)
	{
		double sum2 = 0.0;

		for (int i = 0; i < number_of_correspondences; i++) {
			double * pwa = pws + 3 * i;
			cv::Mat pw = cv::Mat(1, 3, CV_64F, pws + 3 * i);
			double Xc = R.row(0).dot(pw) + t.at<double>(0);
			double Yc = R.row(1).dot(pw) + t.at<double>(1);
			double inv_Zc = 1.0 / (R.row(2).dot(pw) + t.at<double>(2));
			double ue = uc + fu * Xc * inv_Zc;
			double ve = vc + fv * Yc * inv_Zc;
			double u = us[2 * i], v = us[2 * i + 1];
			sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
		}
		return sum2 / number_of_correspondences;
	}

	void PnPSolver::estimate_R_and_t(cv::Mat& R, cv::Mat& t)
	{
		cv::Mat pc0 = cv::Mat::zeros(3, 1,CV_64F);
		cv::Mat pw0 = cv::Mat::zeros(3, 1, CV_64F);
		
		for (int i = 0; i < number_of_correspondences; i++) {
			const double * pc = pcs + 3 * i;
			const double * pw = pws + 3 * i;

			for (int j = 0; j < 3; j++) {
				pc0.at<double>(j) += pc[j];
				pw0.at<double>(j) += pw[j];
			}
		}
		for (int j = 0; j < 3; j++) {
			pc0.at<double>(j) /= number_of_correspondences;
			pw0.at<double>(j) /= number_of_correspondences;
		}

		cv::Mat ABt   = cv::Mat::zeros(3, 3, CV_64F);
		cv::Mat ABt_D = cv::Mat::zeros(3, 1, CV_64F);
		cv::Mat ABt_U = cv::Mat::zeros(3, 3, CV_64F);
		cv::Mat ABt_V = cv::Mat::zeros(3, 3, CV_64F);

		for (int i = 0; i < number_of_correspondences; i++) {
			double * pc = pcs + 3 * i;
			double * pw = pws + 3 * i;

			for (int j = 0; j < 3; j++) {
				ABt.at<double>(j, 0) += (pc[j] - pc0.at<double>(j)) * (pw[0] - pw0.at<double>(0));
				ABt.at<double>(j, 1) += (pc[j] - pc0.at<double>(j)) * (pw[1] - pw0.at<double>(1));
				ABt.at<double>(j, 2) += (pc[j] - pc0.at<double>(j)) * (pw[2] - pw0.at<double>(2));
			}
		}
		cv::SVD::compute(ABt, ABt_D, ABt_U, ABt_V, cv::SVD::MODIFY_A);

		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++) {
				cv::Mat row1 = ABt_U.row(i);
				cv::Mat row2 = ABt_V.row(j);
				R.at<double>(i, j) = row1.dot(row2);
			}


		const double det = cv::determinant(R);
		if (det < 0) {
			R.at<double>(2, 0) *= -1.0;
			R.at<double>(2, 1) *= -1.0;
			R.at<double>(2, 2) *= -1.0;
		}
		
		t.at<double>(0) = pc0.at<double>(0) - R.row(0).dot(pw0.t());
		t.at<double>(1) = pc0.at<double>(1) - R.row(1).dot(pw0.t());
		t.at<double>(2) = pc0.at<double>(2) - R.row(2).dot(pw0.t());
	}
	
	void PnPSolver::solve_for_sign(void)
	{
		if (pcs[2] < 0.0) {
			ccs *= -1.0;
			
			for (int i = 0; i < number_of_correspondences; i++) {
				pcs[3 * i]     = -pcs[3 * i];
				pcs[3 * i + 1] = -pcs[3 * i + 1];
				pcs[3 * i + 2] = -pcs[3 * i + 2];
			}
		}
	}

	double PnPSolver::compute_R_and_t(cv::Mat ut, cv::Mat betas, cv::Mat& R, cv::Mat& t)
	{
		compute_ccs(betas, ut);
		compute_pcs();
		solve_for_sign();
		R = cv::Mat::zeros(3, 3, CV_64F);
		t = cv::Mat::zeros(3, 1, CV_64F);
		estimate_R_and_t(R, t);
		return reprojection_error(R, t);
	}

	// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
	// betas_approx_1 = [B11 B12     B13         B14]

	void PnPSolver::find_betas_approx_1(cv::Mat L_6x10, cv::Mat Rho, cv::Mat& betas)
	{
		cv::Mat L_6x4 = cv::Mat::zeros(6, 4, CV_64F);
		cv::Mat B4    = cv::Mat::zeros(4, 1, CV_64F);

		for (int i = 0; i < 6; i++) {
			L_6x4.at<double>(i, 0) = L_6x10.at<double>(i, 0);
			L_6x4.at<double>(i, 1) = L_6x10.at<double>(i, 1);
			L_6x4.at<double>(i, 2) = L_6x10.at<double>(i, 3);
			L_6x4.at<double>(i, 3) = L_6x10.at<double>(i, 6);
		}
		cv::solve(L_6x4, Rho, B4, cv::DECOMP_SVD);

		
		if (B4.at<double>(0) < 0)
			B4 *= -1.0;
		double val = sqrt(B4.at<double>(0));
		betas = B4/val;
		betas.at<double>(0) = val;
	}

	// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
	// betas_approx_2 = [B11 B12 B22                            ]

	void PnPSolver::find_betas_approx_2(cv::Mat L_6x10, cv::Mat Rho, cv::Mat& betas)
	{
		cv::Mat L_6x3 = cv::Mat::zeros(6, 3, CV_64F);
		cv::Mat B3    = cv::Mat::zeros(3, 1, CV_64F);

		for (int i = 0; i < 6; i++) {
			L_6x3.at<double>(i, 0) = L_6x10.at<double>(i, 0);
			L_6x3.at<double>(i, 1) = L_6x10.at<double>(i, 1);
			L_6x3.at<double>(i, 2) = L_6x10.at<double>(i, 2);
		}
		cv::solve(L_6x3, Rho, B3, cv::DECOMP_SVD);


		double val1 = B3.at<double>(0);
		double val2 = B3.at<double>(1);
		double val3 = B3.at<double>(2);
		if (val1 < 0) {

			betas.at<double>(0) = sqrt(-val1);
			betas.at<double>(1) = (val3 < 0) ? sqrt(-val3) : 0.0;
		}
		else {
			betas.at<double>(0) = sqrt(val1);
			betas.at<double>(1) = (val3 > 0) ? sqrt(val3) : 0.0;
		}

		if (val2 < 0) betas.at<double>(0) = -betas.at<double>(0);

		betas.at<double>(2) = 0.0;
		betas.at<double>(3) = 0.0;
	}

	// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
	// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

	void PnPSolver::find_betas_approx_3(cv::Mat L_6x10, cv::Mat Rho, cv::Mat& betas)
	{
		cv::Mat L_6x5 = cv::Mat::zeros(6, 5, CV_64F);
		cv::Mat B5    = cv::Mat::zeros(5, 1, CV_64F);

		for (int i = 0; i < 6; i++) {
			L_6x5.at<double>(i, 0) = L_6x10.at<double>(i, 0);
			L_6x5.at<double>(i, 1) = L_6x10.at<double>(i, 1);
			L_6x5.at<double>(i, 2) = L_6x10.at<double>(i, 2);
			L_6x5.at<double>(i, 3) = L_6x10.at<double>(i, 3);
			L_6x5.at<double>(i, 4) = L_6x10.at<double>(i, 4);
		}

		cv::solve(L_6x5, Rho, B5, cv::DECOMP_SVD);

		double val1 = B5.at<double>(0);
		double val2 = B5.at<double>(1);
		double val3 = B5.at<double>(2);

		if (val1 < 0) {
			betas.at<double>(0) = sqrt(-val1);
			betas.at<double>(1) = (val3 < 0) ? sqrt(-val3) : 0.0;
		}
		else {
			betas.at<double>(0) = sqrt(val1);
			betas.at<double>(1) = (val3 > 0) ? sqrt(val3) : 0.0;
		}
		if (val2 < 0) betas.at<double>(0) = -betas.at<double>(0);
		betas.at<double>(2) = B5.at<double>(3) / betas.at<double>(0);
		betas.at<double>(3) = 0.0;
	}

	void PnPSolver::compute_L_6x10(cv::Mat Ut, cv::Mat& l_6x10)
	{
		
		double dv[4][6][3];

		for (int i = 0, r = 11; i < 4; i++, r--) {
			int a = 0, b = 1;
			for (int j = 0; j < 6; j++) {

				dv[i][j][0] = Ut.at<double>(r, 3 * a) - Ut.at<double>(r, 3 * b);                
				dv[i][j][1] = Ut.at<double>(r, 3 * a+1) - Ut.at<double>(r, 3 * b+1);			
				dv[i][j][2] = Ut.at<double>(r, 3 * a+2) - Ut.at<double>(r, 3 * b+2);

				b++;
				if (b > 3) {
					a++;
					b = a + 1;
				}
			}
		}

		for (int i = 0; i < 6; i++) {
			int idx = 0;
			l_6x10.at<double>(i, idx++) = dot(dv[0][i], dv[0][i]);
			l_6x10.at<double>(i, idx++) = 2.0f * dot(dv[0][i], dv[1][i]);
			l_6x10.at<double>(i, idx++) = dot(dv[1][i], dv[1][i]);
			l_6x10.at<double>(i, idx++) = 2.0f * dot(dv[0][i], dv[2][i]);
			l_6x10.at<double>(i, idx++) = 2.0f * dot(dv[1][i], dv[2][i]);
			l_6x10.at<double>(i, idx++) = dot(dv[2][i], dv[2][i]);
			l_6x10.at<double>(i, idx++) = 2.0f * dot(dv[0][i], dv[3][i]);
			l_6x10.at<double>(i, idx++) = 2.0f * dot(dv[1][i], dv[3][i]);
			l_6x10.at<double>(i, idx++) = 2.0f * dot(dv[2][i], dv[3][i]);
			l_6x10.at<double>(i, idx++) = dot(dv[3][i], dv[3][i]);
		}
	}

	void PnPSolver::compute_rho(cv::Mat& rho)
	{
		cv::Mat r1 = cws.row(0);
		cv::Mat r2 = cws.row(1);
		cv::Mat r3 = cws.row(2);
		cv::Mat r4 = cws.row(3);
		int idx = 0;
		rho.at<double>(idx++) = dist2(r1, r2);
		rho.at<double>(idx++) = dist2(r1, r3);
		rho.at<double>(idx++) = dist2(r1, r4);
		rho.at<double>(idx++) = dist2(r2, r3);
		rho.at<double>(idx++) = dist2(r2, r4);
		rho.at<double>(idx++) = dist2(r3, r4);
	}

	void PnPSolver::compute_A_and_b_gauss_newton(cv::Mat L_6x10, cv::Mat Rho, cv::Mat betas, Eigen::MatrixXd& A, Eigen::VectorXd& B)
	{
		double a = betas.at<double>(0);
		double b = betas.at<double>(1);
		double c = betas.at<double>(2);
		double d = betas.at<double>(3);

		for (int i = 0; i < 6; i++) {
			cv::Mat B1 = cv::Mat::zeros(4, 1, CV_64FC1);
			B1.at<double>(0) = L_6x10.at<double>(i, 0)*2;
			B1.at<double>(1) = L_6x10.at<double>(i, 1);
			B1.at<double>(2) = L_6x10.at<double>(i, 3);
			B1.at<double>(3) = L_6x10.at<double>(i, 6);

			cv::Mat B2 = cv::Mat::zeros(4, 1, CV_64FC1);
			B2.at<double>(0) = L_6x10.at<double>(i, 1);
			B2.at<double>(1) = L_6x10.at<double>(i, 2) * 2;
			B2.at<double>(2) = L_6x10.at<double>(i, 4);
			B2.at<double>(3) = L_6x10.at<double>(i, 7);

			cv::Mat B3 = cv::Mat::zeros(4, 1, CV_64FC1);
			B3.at<double>(0) = L_6x10.at<double>(i, 3);
			B3.at<double>(1) = L_6x10.at<double>(i, 4);
			B3.at<double>(2) = L_6x10.at<double>(i, 5) * 2;
			B3.at<double>(3) = L_6x10.at<double>(i, 8);

			cv::Mat B4 = cv::Mat::zeros(4, 1, CV_64FC1);
			B4.at<double>(0) = L_6x10.at<double>(i, 6);
			B4.at<double>(1) = L_6x10.at<double>(i, 7);
			B4.at<double>(2) = L_6x10.at<double>(i, 8);
			B4.at<double>(3) = L_6x10.at<double>(i, 9) * 2;
			
			A(i, 0) = B1.dot(betas);
			A(i, 1) = B2.dot(betas);
			A(i, 2) = B3.dot(betas);
			A(i, 3) = B4.dot(betas);

			B[i] = Rho.at<double>(i, 0)-
				(
				L_6x10.at<double>(i, 0)*a*a + 
				L_6x10.at<double>(i, 1)*a*b +
				L_6x10.at<double>(i, 2)*b*b +
				L_6x10.at<double>(i, 3)*a*c +
				L_6x10.at<double>(i, 4)*b*c +
				L_6x10.at<double>(i, 5)*c*c +
				L_6x10.at<double>(i, 6)*a*d +
				L_6x10.at<double>(i, 7)*b*d +
				L_6x10.at<double>(i, 8)*c*d +
				L_6x10.at<double>(i, 9)*d*d
				);
		}
	}

	void PnPSolver::gauss_newton(cv::Mat L_6x10, cv::Mat Rho, cv::Mat betas)
	{
		const int iterations_number = 5;

		Eigen::MatrixXd A = Eigen::MatrixXd(6, 4);
		Eigen::VectorXd B = Eigen::VectorXd(6);

		/*cv::Mat A = cv::Mat::zeros(6, 4, CV_64F);
		cv::Mat B = cv::Mat::zeros(6, 1, CV_64F);
		cv::Mat X = cv::Mat::zeros(4, 1, CV_64F);*/

		for (int k = 0; k < iterations_number; k++) {
			compute_A_and_b_gauss_newton(L_6x10, Rho, betas, A, B);
			Eigen::VectorXd X = A.colPivHouseholderQr().solve(B);
			for (int i = 0; i < 4; i++)
				betas.at<double>(i) = X(i);
		}
	}

	/*
	void PnPSolver::relative_error(double & rot_err, double & transl_err,
		const double Rtrue[3][3], const double ttrue[3],
		const double Rest[3][3], const double test[3])
	{
		double qtrue[4], qest[4];

		mat_to_quat(Rtrue, qtrue);
		mat_to_quat(Rest, qest);

		double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
			(qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
			(qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
			(qtrue[3] - qest[3]) * (qtrue[3] - qest[3])) /
			sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

		double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
			(qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
			(qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
			(qtrue[3] + qest[3]) * (qtrue[3] + qest[3])) /
			sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

		rot_err = std::min(rot_err1, rot_err2);

		transl_err =
			sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
			(ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
				(ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
			sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
	}

	void PnPSolver::mat_to_quat(const double R[3][3], double q[4])
	{
		double tr = R[0][0] + R[1][1] + R[2][2];
		double n4;

		if (tr > 0.0f) {
			q[0] = R[1][2] - R[2][1];
			q[1] = R[2][0] - R[0][2];
			q[2] = R[0][1] - R[1][0];
			q[3] = tr + 1.0f;
			n4 = q[3];
		}
		else if ((R[0][0] > R[1][1]) && (R[0][0] > R[2][2])) {
			q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
			q[1] = R[1][0] + R[0][1];
			q[2] = R[2][0] + R[0][2];
			q[3] = R[1][2] - R[2][1];
			n4 = q[0];
		}
		else if (R[1][1] > R[2][2]) {
			q[0] = R[1][0] + R[0][1];
			q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
			q[2] = R[2][1] + R[1][2];
			q[3] = R[2][0] - R[0][2];
			n4 = q[1];
		}
		else {
			q[0] = R[2][0] + R[0][2];
			q[1] = R[2][1] + R[1][2];
			q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
			q[3] = R[0][1] - R[1][0];
			n4 = q[2];
		}
		double scale = 0.5f / double(sqrt(n4));

		q[0] *= scale;
		q[1] *= scale;
		q[2] *= scale;
		q[3] *= scale;
	}
	*/
}
