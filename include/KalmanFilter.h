
#ifndef DYNAMIC_ESTIMATOR_KALMAN_FILTER_H
#define DYNAMIC_ESTIMATOR_KALMAN_FILTER_H

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
//namespace EdgeSLAM {
    class KalmanFilter
    {
    public:
        KalmanFilter();
        KalmanFilter(int _nStates, int _nMeasurements, int _nInputs, double _dt);
        virtual ~KalmanFilter();

    public:
        void initKalmanFilter();
        void Predict(cv::Mat& P);
        void Correct(cv::Mat& P);
        void updateKalmanFilter(cv::Mat& translation_estimated, cv::Mat& rotation_estimated);
        void fillMeasurements(const cv::Mat& translation_measured, const cv::Mat& rotation_measured);
    private:
        cv::Mat euler2rot(const cv::Mat& euler);
        cv::Mat rot2euler(const cv::Mat& rotationMatrix);
    private:
        cv::KalmanFilter mKalmanFilter;
        cv::Mat measurements;
        int nStates;
        int nMeasurements;
        int nInputs;
        double dt;
    };

//}

#endif /* PNPPROBLEM_H_ */
