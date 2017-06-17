#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */


	ekf_.P_ = MatrixXd (4,4);
	ekf_.P_<<	10,0,0,0,
						0,10,0,0,
						0,0,100,0,
						0,0,0,100;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
	long long current_timestamp = measurement_pack.timestamp_;
	float dt = (current_timestamp - previous_timestamp_)/1000000.0;
	previous_timestamp_ = measurement_pack.timestamp_;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
			* Initialize the state ekf_.x_ with the first measurement.
			* Create the covariance matrix.
			* Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			/**
			Convert radar from polar to cartesian coordinates and initialize state.
			*/
			float ro = measurement_pack.raw_measurements_[0];
			float theta = measurement_pack.raw_measurements_[1];
			float ro_dot = measurement_pack.raw_measurements_[2];
			float px = ro * cos(theta);
			float py = ro * sin(theta);
			float vx = ro_dot * cos(theta);
			float vy = ro_dot * sin(theta);

			ekf_.x_ << px,py,0,0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
			float px = measurement_pack.raw_measurements_[0];
			float py = measurement_pack.raw_measurements_[1];
			ekf_.x_ << px,px,0,0;
		}

		// Create the covariance matrix.
		//ekf_.Q_=update_process_covariance(dt);

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

	update_state_transition(dt);
	update_process_covariance(dt);

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		Tools tools=Tools();
		ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
		ekf_.R_ = R_radar_;
		ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
		// Laser updates
		ekf_.H_ = MatrixXd(2,4);
		ekf_.H_ <<	1,0,0,0,
								0,1,0,0;
		ekf_.R_ = R_laser_;
		ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

void FusionEKF::update_process_covariance(float dt){
	float dt2 = dt*dt;
	float noise_ax = 9, noise_ay = 9;

	MatrixXd G = MatrixXd(4,2);
	G << dt2/2, 0,
			0, dt2/2,
			dt, 0,
			0, dt;
	MatrixXd Gtrans = G.transpose();

	MatrixXd process_noise_covariance = MatrixXd(2,2);
	process_noise_covariance << noise_ax,0,
															0,noise_ay;
	ekf_.Q_ = G * (process_noise_covariance) * Gtrans;
}

void FusionEKF::update_state_transition(float dt){
	MatrixXd temp(4,4);
	temp <<	1,0,dt,0,
					0,1,0,dt,
					0,0,1,0,
					0,0,0,1;
	ekf_.F_ = temp;
}
