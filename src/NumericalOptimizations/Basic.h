#pragma once
#include "ObjectiveFunctions/Total.h"
#include <atomic>
#include <shared_mutex>
#include <igl/flip_avoiding_line_search.h>
#include <Eigen/SparseCholesky>
#include <fstream>

class Minimizer
{
public:
	Cuda::Array<double> X, p, curr_x, v_adam, s_adam;

	Minimizer(const int solverID);
	void run();
	void run_new();
	void RunSymmetricDirichletGradient();
	void run_one_iteration();
	void run_one_iteration_new();
	void stop();
	void get_data(
		Eigen::MatrixXd& X, 
		Eigen::MatrixXd& center, 
		Eigen::VectorXd& radius,
		Eigen::MatrixXd& norm);
	void init(
		std::shared_ptr<TotalObjective> Tobjective,
		const Eigen::VectorXd& X0,
		const Eigen::VectorXd& norm0,
		const Eigen::VectorXd& center0,
		const Eigen::VectorXd& Radius0,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixXd& V);
	void upload_x(const Eigen::VectorXd& X0);

	// Pointer to the energy class
	std::shared_ptr<TotalObjective> totalObjective;
	// Activity flags
	std::atomic_bool is_running = {false};
	std::atomic_bool progressed = {false};
	std::atomic_bool isGradientNeeded = { false };


	// Synchronization functions used by the wrapper
	void wait_for_parameter_update_slot();
	void release_parameter_update_slot();

	// External (interface) and internal working mesh
	Eigen::VectorXd ext_x, ext_center, ext_radius, ext_norm;
	Eigen::MatrixX3i F;
	Eigen::MatrixXd V;
	Cuda::OptimizerType Optimizer_type;
	double timer_curr=0, timer_sum = 0, timer_avg = 0;

	OptimizationUtils::LineSearch lineSearch_type;
	double constantStep_LineSearch;
	inline int getNumiter() {
		return this->numIteration;
	}
	void update_lambda();
	bool isAutoLambdaRunning = true;
	bool isUpdateLambdaWhenConverge = false;
	int autoLambda_from = 100, autoLambda_count = 40, autoLambda_jump = 70;
	double init_step_size = 1;
	unsigned int linesearch_numiterations = 0;
	unsigned int linesearch_StopCounter = 0;
private:
	// Give the wrapper a chance to intersect gracefully
	void give_parameter_update_slot();
	// Updating the data after a step has been done
	void update_external_data();
	double currentEnergy;
	unsigned int numIteration = 0;
	int solverID;
	void linesearch();
	void value_linesearch();
	void gradNorm_linesearch();
	void constant_linesearch();
	double step_size;

	// Mutex stuff
	std::unique_ptr<std::shared_timed_mutex> data_mutex;
	std::unique_ptr<std::mutex> parameters_mutex;
	std::unique_ptr<std::condition_variable> param_cv;
	// Synchronization structures
	std::atomic_bool params_ready_to_update = { false };
	std::atomic_bool wait_for_param_update = { false };
	std::atomic_bool halt = { false };
};