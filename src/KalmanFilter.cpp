#include "KalmanFilter.h"
//namespace EdgeSLAM {
    KalmanFilter::KalmanFilter() {}
    KalmanFilter::KalmanFilter(int _nStates, int _nMeasurements, int _nInputs, double _dt) :
        nStates(_nStates), nMeasurements(_nMeasurements), nInputs(_nInputs), dt(_dt)
    {
        initKalmanFilter();
    }
    KalmanFilter::~KalmanFilter() {}

    void KalmanFilter::initKalmanFilter()
    {
        mKalmanFilter.init(nStates, nMeasurements, nInputs, CV_64F);                 // init Kalman Filter
        setIdentity(mKalmanFilter.processNoiseCov, cv::Scalar::all(1e-5));       // set process noise
        setIdentity(mKalmanFilter.measurementNoiseCov, cv::Scalar::all(1e-2));   // set measurement noise
        setIdentity(mKalmanFilter.errorCovPost, cv::Scalar::all(1));             // error covariance

        /** DYNAMIC MODEL **/

        //  [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
        //  [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
        //  [0 0 1  0  0 dt   0   0 dt2 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
        //  [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
        //  [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
        //  [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
        //  [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]

        // position
        mKalmanFilter.transitionMatrix.at<double>(0, 3) = dt;
        mKalmanFilter.transitionMatrix.at<double>(1, 4) = dt;
        mKalmanFilter.transitionMatrix.at<double>(2, 5) = dt;
        mKalmanFilter.transitionMatrix.at<double>(3, 6) = dt;
        mKalmanFilter.transitionMatrix.at<double>(4, 7) = dt;
        mKalmanFilter.transitionMatrix.at<double>(5, 8) = dt;
        mKalmanFilter.transitionMatrix.at<double>(0, 6) = 0.5 * pow(dt, 2);
        mKalmanFilter.transitionMatrix.at<double>(1, 7) = 0.5 * pow(dt, 2);
        mKalmanFilter.transitionMatrix.at<double>(2, 8) = 0.5 * pow(dt, 2);

        // orientation
        mKalmanFilter.transitionMatrix.at<double>(9, 12) = dt;
        mKalmanFilter.transitionMatrix.at<double>(10, 13) = dt;
        mKalmanFilter.transitionMatrix.at<double>(11, 14) = dt;
        mKalmanFilter.transitionMatrix.at<double>(12, 15) = dt;
        mKalmanFilter.transitionMatrix.at<double>(13, 16) = dt;
        mKalmanFilter.transitionMatrix.at<double>(14, 17) = dt;
        mKalmanFilter.transitionMatrix.at<double>(9, 15) = 0.5 * pow(dt, 2);
        mKalmanFilter.transitionMatrix.at<double>(10, 16) = 0.5 * pow(dt, 2);
        mKalmanFilter.transitionMatrix.at<double>(11, 17) = 0.5 * pow(dt, 2);


        /** MEASUREMENT MODEL **/

        //  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        //  [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        //  [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        //  [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
        //  [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
        //  [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]

        mKalmanFilter.measurementMatrix.at<double>(0, 0) = 1;  // x
        mKalmanFilter.measurementMatrix.at<double>(1, 1) = 1;  // y
        mKalmanFilter.measurementMatrix.at<double>(2, 2) = 1;  // z
        mKalmanFilter.measurementMatrix.at<double>(3, 9) = 1;  // roll
        mKalmanFilter.measurementMatrix.at<double>(4, 10) = 1; // pitch
        mKalmanFilter.measurementMatrix.at<double>(5, 11) = 1; // yaw
    }
    void KalmanFilter::Predict(cv::Mat& P) {
        cv::Mat prediction = mKalmanFilter.predict();

        // Estimated translation
        cv::Mat t = cv::Mat::zeros(3, 1, CV_64FC1);
        t.at<double>(0) = prediction.at<double>(0);
        t.at<double>(1) = prediction.at<double>(1);
        t.at<double>(2) = prediction.at<double>(2);

        // Estimated euler angles
        cv::Mat eulers_estimated(3, 1, CV_64F);
        eulers_estimated.at<double>(0) = prediction.at<double>(9);
        eulers_estimated.at<double>(1) = prediction.at<double>(10);
        eulers_estimated.at<double>(2) = prediction.at<double>(11);

        // Convert estimated quaternion to rotation matrix
        cv::Mat R = euler2rot(eulers_estimated);
        P = cv::Mat::eye(4, 4, CV_64FC1);
        R.copyTo(P.rowRange(0, 3).colRange(0, 3));
        t.copyTo(P.rowRange(0, 3).col(3));
        P.convertTo(P, CV_32FC1);
    }
    void KalmanFilter::Correct(cv::Mat& P) {
        //32실수형으로 리턴
        cv::Mat estimated = mKalmanFilter.correct(measurements);

        // Estimated translation
        cv::Mat translation_estimated(3, 1, CV_64FC1);
        translation_estimated.at<double>(0) = estimated.at<double>(0);
        translation_estimated.at<double>(1) = estimated.at<double>(1);
        translation_estimated.at<double>(2) = estimated.at<double>(2);

        // Estimated euler angles
        cv::Mat eulers_estimated(3, 1, CV_64F);
        eulers_estimated.at<double>(0) = estimated.at<double>(9);
        eulers_estimated.at<double>(1) = estimated.at<double>(10);
        eulers_estimated.at<double>(2) = estimated.at<double>(11);

        // Convert estimated quaternion to rotation matrix
        cv::Mat rotation_estimated = euler2rot(eulers_estimated);

        // Convert estimated quaternion to rotation matrix
        
        P = cv::Mat::eye(4, 4, CV_64FC1);
        rotation_estimated.copyTo(P.rowRange(0, 3).colRange(0, 3));
        translation_estimated.copyTo(P.rowRange(0, 3).col(3));
        P.convertTo(P, CV_32FC1);
    }
    void KalmanFilter::updateKalmanFilter(cv::Mat& translation_estimated, cv::Mat& rotation_estimated)
    {
        //translation_estimated.convertTo(translation_estimated, CV_64FC1);
        //rotation_estimated.convertTo(rotation_estimated, CV_64FC1);

        // First predict, to update the internal statePre variable
        cv::Mat prediction = mKalmanFilter.predict();

        // The "correct" phase that is going to use the predicted value and our measurement
        cv::Mat estimated = mKalmanFilter.correct(measurements);

        // Estimated translation
        translation_estimated.at<double>(0) = estimated.at<double>(0);
        translation_estimated.at<double>(1) = estimated.at<double>(1);
        translation_estimated.at<double>(2) = estimated.at<double>(2);

        // Estimated euler angles
        cv::Mat eulers_estimated(3, 1, CV_64F);
        eulers_estimated.at<double>(0) = estimated.at<double>(9);
        eulers_estimated.at<double>(1) = estimated.at<double>(10);
        eulers_estimated.at<double>(2) = estimated.at<double>(11);

        // Convert estimated quaternion to rotation matrix
        rotation_estimated = euler2rot(eulers_estimated);

        //translation_estimated.convertTo(translation_estimated, CV_32FC1);
        //rotation_estimated.convertTo(rotation_estimated, CV_32FC1);
    }

    void KalmanFilter::fillMeasurements(const cv::Mat& translation_measured, const cv::Mat& rotation_measured)
    {
        //translation_measured.convertTo(translation_measured, CV_64FC1);
        //rotation_measured.convertTo(rotation_measured, CV_64FC1);

        measurements = cv::Mat::zeros(nMeasurements, 1, CV_64FC1);
        // Convert rotation matrix to euler angles
        cv::Mat measured_eulers(3, 1, CV_64F);
        measured_eulers = rot2euler(rotation_measured);

        // Set measurement to predict
        measurements.at<double>(0) = translation_measured.at<double>(0); // x
        measurements.at<double>(1) = translation_measured.at<double>(1); // y
        measurements.at<double>(2) = translation_measured.at<double>(2); // z
        measurements.at<double>(3) = measured_eulers.at<double>(0);      // roll
        measurements.at<double>(4) = measured_eulers.at<double>(1);      // pitch
        measurements.at<double>(5) = measured_eulers.at<double>(2);      // yaw
    }

    cv::Mat KalmanFilter::rot2euler(const cv::Mat& rotationMatrix)
    {
        cv::Mat euler(3, 1, CV_64F);

        double m00 = rotationMatrix.at<double>(0, 0);
        double m02 = rotationMatrix.at<double>(0, 2);
        double m10 = rotationMatrix.at<double>(1, 0);
        double m11 = rotationMatrix.at<double>(1, 1);
        double m12 = rotationMatrix.at<double>(1, 2);
        double m20 = rotationMatrix.at<double>(2, 0);
        double m22 = rotationMatrix.at<double>(2, 2);

        double bank, attitude, heading;

        // Assuming the angles are in radians.
        if (m10 > 0.998) { // singularity at north pole
            bank = 0;
            attitude = CV_PI / 2;
            heading = atan2(m02, m22);
        }
        else if (m10 < -0.998) { // singularity at south pole
            bank = 0;
            attitude = -CV_PI / 2;
            heading = atan2(m02, m22);
        }
        else
        {
            bank = atan2(-m12, m11);
            attitude = asin(m10);
            heading = atan2(-m20, m00);
        }

        euler.at<double>(0) = bank;
        euler.at<double>(1) = attitude;
        euler.at<double>(2) = heading;

        return euler;
    }

    cv::Mat KalmanFilter::euler2rot(const cv::Mat& euler)
    {
        cv::Mat rotationMatrix(3, 3, CV_64F);

        double bank = euler.at<double>(0);
        double attitude = euler.at<double>(1);
        double heading = euler.at<double>(2);

        // Assuming the angles are in radians.
        double ch = cos(heading);
        double sh = sin(heading);
        double ca = cos(attitude);
        double sa = sin(attitude);
        double cb = cos(bank);
        double sb = sin(bank);

        double m00, m01, m02, m10, m11, m12, m20, m21, m22;

        m00 = ch * ca;
        m01 = sh * sb - ch * sa * cb;
        m02 = ch * sa * sb + sh * cb;
        m10 = sa;
        m11 = ca * cb;
        m12 = -ca * sb;
        m20 = -sh * ca;
        m21 = sh * sa * cb + ch * sb;
        m22 = -sh * sa * sb + ch * cb;

        rotationMatrix.at<double>(0, 0) = m00;
        rotationMatrix.at<double>(0, 1) = m01;
        rotationMatrix.at<double>(0, 2) = m02;
        rotationMatrix.at<double>(1, 0) = m10;
        rotationMatrix.at<double>(1, 1) = m11;
        rotationMatrix.at<double>(1, 2) = m12;
        rotationMatrix.at<double>(2, 0) = m20;
        rotationMatrix.at<double>(2, 1) = m21;
        rotationMatrix.at<double>(2, 2) = m22;

        return rotationMatrix;
    }
//}
