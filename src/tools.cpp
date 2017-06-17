#include <iostream>
#include "tools.h"
#include <assert.h>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

	if (estimations.size()!=ground_truth.size() || estimations.size()==0){
		std::cout << "invalid data to calculate the RMSE" << std::endl;
		return rmse;
	}

	//accumulate squared residuals
	for(int i=0; i < estimations.size(); i++){
		VectorXd residual = estimations[i] - ground_truth[i];
		residual = residual.array() * residual.array();
		rmse += residual;
	}
	rmse = rmse / estimations.size();
	rmse = rmse.array().sqrt();
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);
	Hj << 1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0;
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//check division by zero
	float ro_sq = pow(px,2) + pow(py,2);
	float ro = sqrt(ro_sq);
	if(fabs(ro_sq) < 1e-13){
			cout << "division by zero in jacobian matrix" << endl;
			return Hj;
	}

	//compute the Jacobian matrix
	Hj(0,0) = px/ro;
	Hj(0,1) = py/ro;
	Hj(0,2) = 0;
	Hj(0,3) = 0;
	Hj(1,0) = -py / ro_sq;
	Hj(1,1) = px / ro_sq;
	Hj(1,2) = 0;
	Hj(1,3) = 0;
	Hj(2,0) = ( py* (vx*py-vy*px) )/pow(ro_sq,1.5);
	Hj(2,1) = ( px* (vy*px-vx*py) )/pow(ro_sq,1.5);
	Hj(2,2) = px/ro;
	Hj(2,3) = py/ro;

	return Hj;
}
