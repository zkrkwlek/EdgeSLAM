#ifndef EDGE_SLAM_PNPSOLVER_H
#define EDGE_SLAM_LOOP_CLOSER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace EdgeSLAM {
	class Frame;
	class MapPoint;
	class PnPSolver {
	public:
		PnPSolver(Frame *F,std::vector<MapPoint*> vpMapPointMatches);
		virtual ~PnPSolver();

		void SetRansacParameters(double probability = 0.99, int minInliers = 8, int maxIterations = 300, int minSet = 4, double epsilon = 0.4,
			double th2 = 5.991);

		cv::Mat find(std::vector<bool> &vbInliers, int &nInliers);

		cv::Mat iterate(int nIterations, bool &bNoMore, std::vector<bool> &vbInliers, int &nInliers);

	private:
		void CheckInliers();
		bool Refine();

		// Functions from the original EPnP code
		void set_maximum_number_of_correspondences(const int n);
		void reset_correspondences(void);
		void add_correspondence(const double X, const double Y, const double Z,
			const double u, const double v);

		double compute_pose(cv::Mat& R, cv::Mat& t);

		double reprojection_error(cv::Mat R, cv::Mat t);

		void choose_control_points(void);
		void compute_barycentric_coordinates(void);
		void fill_M(cv::Mat& M, const int row, const double * as, const double u, const double v);
		void compute_ccs(cv::Mat betas, cv::Mat ut);
		void compute_pcs(void);
		void solve_for_sign(void);

		void find_betas_approx_1(cv::Mat L_6x10, cv::Mat Rho, cv::Mat& betas);
		void find_betas_approx_2(cv::Mat L_6x10, cv::Mat Rho, cv::Mat& betas);
		void find_betas_approx_3(cv::Mat L_6x10, cv::Mat Rho, cv::Mat& betas);

		double dot(const double * v1, const double * v2);
		double dist2(cv::Mat p1, cv::Mat p2);

		void compute_rho(cv::Mat& rho);
		void compute_L_6x10(cv::Mat Ut,cv::Mat& l_6x10);

		void gauss_newton(cv::Mat L_6x10, cv::Mat Rho, cv::Mat& betas);
		void compute_A_and_b_gauss_newton(cv::Mat L_6x10, cv::Mat Rho, cv::Mat betas, Eigen::MatrixXd& A, Eigen::VectorXd& b);

		double compute_R_and_t(cv::Mat ut, cv::Mat betas, cv::Mat& R, cv::Mat& t);

		void estimate_R_and_t(cv::Mat& R, cv::Mat& t);
		
		//void relative_error(double & rot_err, double & transl_err, cv::Mat Rtrue, cv::Mat ttrue, cv::Mat Rest, cv::Mat test);
		//void mat_to_quat(const double R[3][3], double q[4]);


		double uc, vc, fu, fv;

		double * pws, *us, *alphas, *pcs;
		int maximum_number_of_correspondences;
		int number_of_correspondences;

		//double cws[4][3], ccs[4][3];
		cv::Mat cws, ccs;//4x3
		double cws_determinant;

		std::vector<MapPoint*> mvpMapPointMatches;

		// 2D Points
		std::vector<cv::Point2d> mvP2D;
		std::vector<double> mvSigma2;

		// 3D Points
		std::vector<cv::Point3d> mvP3Dw;

		// Index in Frame
		std::vector<size_t> mvKeyPointIndices;

		// Current Estimation
		cv::Mat mRi;
		cv::Mat mti;
		cv::Mat mTcwi;
		std::vector<bool> mvbInliersi;
		int mnInliersi;

		// Current Ransac State
		int mnIterations;
		std::vector<bool> mvbBestInliers;
		int mnBestInliers;
		cv::Mat mBestTcw;

		// Refined
		cv::Mat mRefinedTcw;
		std::vector<bool> mvbRefinedInliers;
		int mnRefinedInliers;

		// Number of Correspondences
		int N;

		// Indices for random selection [0 .. N-1]
		std::vector<size_t> mvAllIndices;

		// RANSAC probability
		double mRansacProb;

		// RANSAC min inliers
		int mRansacMinInliers;

		// RANSAC max iterations
		int mRansacMaxIts;

		// RANSAC expected inliers/total ratio
		double mRansacEpsilon;

		// RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
		double mRansacTh;

		// RANSAC Minimun Set used at each iteration
		int mRansacMinSet;

		// Max square error associated with scale level. Max error = th*th*sigma(level)*sigma(level)
		std::vector<double> mvMaxError;
	};
}
#endif


