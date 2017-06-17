#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	x_ = F_ * x_;
	P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd y = z - H_*x_;
	update_x_P(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	VectorXd y = UpdateError(z);
	update_x_P(y);
}

VectorXd KalmanFilter::UpdateError(const VectorXd &z) {
	float px = x_[0];
	float py = x_[1];
	float vx = x_[2];
	float vy = x_[3];
	float ro = sqrt(pow(px,2) + pow(py,2));

	if(fabs(ro) < 1e-30){
		std::cout << "Update error impossible, Devision by zero" << std::endl;
		VectorXd y = VectorXd(z.size());
		for (int i = 0; i < z.size() ; i++)
			y[i] = 1;
		return y;
	}

	VectorXd Hx = VectorXd(3);
	Hx << ro ,
			atan2(py,px) ,
			(px*vx + py*vy)/ro;
	return (z - Hx);
}

void KalmanFilter::update_x_P(VectorXd y){
	MatrixXd S = (H_ * P_ * (H_.transpose()) ) + R_;
	MatrixXd K = P_ * (H_.transpose()) * (S.inverse());

	x_ = x_ + (K * y);
	MatrixXd I = MatrixXd::Identity(4, 4);
	P_ = (I - (K * H_) )*P_;
}
