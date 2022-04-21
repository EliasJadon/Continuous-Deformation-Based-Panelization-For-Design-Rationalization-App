#include "NumericalOptimizations/Basic.h"
#include "ObjectiveFunctions/Panels/AuxPlanar.h"
#include "ObjectiveFunctions/Panels/Planar.h"
#include "ObjectiveFunctions/Panels/AuxSphere.h"
#include "ObjectiveFunctions/Deformation/SymmetricDirichlet.h"

#define BETA1_ADAM 0.90f
#define BETA2_ADAM 0.9990f
#define EPSILON_ADAM 1e-8
#define MAX_STEP_SIZE_ITER 50
//#define PRINT_CLI

using namespace NumericalOptimizations;


Basic::Basic(const int solverID)
	:
	solverID(solverID),
	parameters_mutex(std::make_unique<std::mutex>()),
	data_mutex(std::make_unique<std::shared_timed_mutex>()),
	param_cv(std::make_unique<std::condition_variable>())
{ }

void Basic::init(
	std::shared_ptr<ObjectiveFunctions::Total> Tobjective,
	const Eigen::VectorXd& X0,
	const Eigen::VectorXd& norm0,
	const Eigen::VectorXd& center0,
	const Eigen::VectorXd& Radius0,
	const Eigen::MatrixXi& F, 
	const Eigen::MatrixXd& V) 
{
	assert(X0.rows()			== (3 * V.rows()) && "X0 size is illegal!");
	assert(norm0.rows()			== (3 * F.rows()) && "norm0 size is illegal!");
	assert(center0.rows()		== (3 * F.rows()) && "center0 size is illegal!");
	assert(Radius0.rows()		== (1 * F.rows()) && "Radius0 size is illegal!");
	
	this->F = F;
	this->V = V;
	this->ext_x = X0;
	this->ext_center = center0;
	this->ext_norm = norm0;
	this->ext_radius = Radius0;
	this->constantStep_LineSearch = 0.01;
	this->totalObjective = Tobjective;
	
	unsigned int size = 3 * V.rows() + 7 * F.rows();

	Cuda::AllocateMemory(X, size);
	Cuda::AllocateMemory(p, size);
	Cuda::AllocateMemory(curr_x, size);
	Cuda::AllocateMemory(v_adam, size);
	Cuda::AllocateMemory(s_adam, size);
	for (int i = 0; i < size; i++) {
		v_adam.host_arr[i] = 0;
		s_adam.host_arr[i] = 0;
		p.host_arr[i] = 0;
	}
	
	for (int i = 0; i < 3 * V.rows(); i++)
		X.host_arr[0 * V.rows() + 0 * F.rows() + i] = X0[i];
	for (int i = 0; i < 3 * F.rows(); i++)
		X.host_arr[3 * V.rows() + 0 * F.rows() + i] = norm0[i];
	for (int i = 0; i < 3 * F.rows(); i++)
		X.host_arr[3 * V.rows() + 3 * F.rows() + i] = center0[i];
	for (int i = 0; i < F.rows(); i++)
		X.host_arr[3 * V.rows() + 6 * F.rows() + i] = Radius0[i];
	for (int i = 0; i < totalObjective->grad.size; i++) {
		curr_x.host_arr[i] = X.host_arr[i];
	}
}

void Basic::upload_x(const Eigen::VectorXd& X0)
{
	assert(X0.rows() == (3 * V.rows()) && "X0 size is illegal!");
	for (int i = 0; i < 3 * V.rows(); i++)
		X.host_arr[0 * V.rows() + 0 * F.rows() + i] = X0[i];
	for (int i = 0; i < totalObjective->grad.size; i++)
		curr_x.host_arr[i] = X.host_arr[i];
	update_external_data();
}

void Basic::run()
{
	is_running = true;
	halt = false;
	while (!halt)
		run_one_iteration();
	is_running = false;
	std::cout << ">> solver " + std::to_string(solverID) + " stopped" << std::endl;
}

void Basic::run_new()
{
	is_running = true;
	halt = false;
	while (!halt)
		run_one_iteration_new();
	is_running = false;
	std::cout << ">> solver " + std::to_string(solverID) + " stopped" << std::endl;
}

void Basic::RunSymmetricDirichletGradient() {
	halt = false;
	while (!halt) {
		std::shared_ptr<ObjectiveFunctions::Deformation::SymmetricDirichlet> SD = std::dynamic_pointer_cast<ObjectiveFunctions::Deformation::SymmetricDirichlet>(totalObjective->objectiveList[4]);
		if (isGradientNeeded) {
			if (SD->w != 0)
				SD->gradient(X, false);
			isGradientNeeded = false;
		}
	}
}

void Basic::update_lambda()
{
	std::shared_ptr<ObjectiveFunctions::Panels::AuxSphere> ASH = 
		std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxSphere>(totalObjective->objectiveList[0]);
	std::shared_ptr<ObjectiveFunctions::Panels::AuxPlanar> ABN = 
		std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxPlanar>(totalObjective->objectiveList[1]);
	std::shared_ptr<ObjectiveFunctions::Panels::Planar> BN =
		std::dynamic_pointer_cast<ObjectiveFunctions::Panels::Planar>(totalObjective->objectiveList[2]);
	
	if (isAutoLambdaRunning && numIteration >= autoLambda_from && !(numIteration % autoLambda_jump))
	{
		const double target = pow(2, -autoLambda_count);
		ASH->Dec_SigmoidParameter(target);
		ABN->Dec_SigmoidParameter(target);
		BN->Dec_SigmoidParameter(target);
	}
}

void Basic::run_one_iteration()
{
	OptimizationUtils::Timer t(&timer_sum, &timer_curr);
	timer_avg = timer_sum / ++numIteration;
	update_lambda();

	totalObjective->gradient(X, false);
	if (Optimizer_type == Cuda::OptimizerType::Adam) {
		for (int i = 0; i < totalObjective->grad.size; i++) {
			v_adam.host_arr[i] = BETA1_ADAM * v_adam.host_arr[i] + (1 - BETA1_ADAM) * totalObjective->grad.host_arr[i];
			s_adam.host_arr[i] = BETA2_ADAM * s_adam.host_arr[i] + (1 - BETA2_ADAM) * pow(totalObjective->grad.host_arr[i], 2);
			p.host_arr[i] = -v_adam.host_arr[i] / (sqrt(s_adam.host_arr[i]) + EPSILON_ADAM);
		}
	}
	else if (Optimizer_type == Cuda::OptimizerType::Gradient_Descent) {
		for (int i = 0; i < totalObjective->grad.size; i++) {
			p.host_arr[i] = -totalObjective->grad.host_arr[i];
		}
	}
	currentEnergy = totalObjective->value(X, true);
	linesearch();
	update_external_data();
}

void Basic::run_one_iteration_new()
{
	OptimizationUtils::Timer t(&timer_sum, &timer_curr);
	//calculate SD gradient in advance
	isGradientNeeded = true;
	std::shared_ptr<ObjectiveFunctions::Deformation::SymmetricDirichlet> SD = std::dynamic_pointer_cast<ObjectiveFunctions::Deformation::SymmetricDirichlet>(totalObjective->objectiveList[4]);
	
	timer_avg = timer_sum / ++numIteration;
	update_lambda();

	//////////////////////////
	//Get the first part of the gradients
	for (int i = 0; i < totalObjective->grad.size; i++)
		totalObjective->grad.host_arr[i] = 0;
	for (auto& obj : totalObjective->objectiveList) {
		std::shared_ptr<ObjectiveFunctions::Deformation::SymmetricDirichlet> SD = std::dynamic_pointer_cast<ObjectiveFunctions::Deformation::SymmetricDirichlet>(obj);
		if (obj->w != 0 && SD == NULL) {
			obj->gradient(X, false);
			for (int i = 0; i < totalObjective->grad.size; i++)
				totalObjective->grad.host_arr[i] += obj->w * obj->grad.host_arr[i];
		}
	}
	//////////////////////////
	//caculate current value
#ifdef PRINT_CLI
		std::cout << numIteration << ", ";
		currentEnergy = totalObjective->value_print(X, true);
#else
		currentEnergy = totalObjective->value(X, true);
#endif
	//////////////////////////
	///get the second part of the gradient
	while (isGradientNeeded);
	if (SD->w != 0) {
		for (int i = 0; i < totalObjective->grad.size; i++)
			totalObjective->grad.host_arr[i] += SD->w * SD->grad.host_arr[i];
	}
	//////////////////////////

	if (Optimizer_type == Cuda::OptimizerType::Adam) {
		for (int i = 0; i < totalObjective->grad.size; i++) {
			v_adam.host_arr[i] = BETA1_ADAM * v_adam.host_arr[i] + (1 - BETA1_ADAM) * totalObjective->grad.host_arr[i];
			s_adam.host_arr[i] = BETA2_ADAM * s_adam.host_arr[i] + (1 - BETA2_ADAM) * pow(totalObjective->grad.host_arr[i], 2);
			p.host_arr[i] = -v_adam.host_arr[i] / (sqrt(s_adam.host_arr[i]) + EPSILON_ADAM);
		}
	}
	else if (Optimizer_type == Cuda::OptimizerType::Gradient_Descent) {
		for (int i = 0; i < totalObjective->grad.size; i++) {
			p.host_arr[i] = -totalObjective->grad.host_arr[i];
		}
	}
	linesearch();
	update_external_data();
}



void Basic::linesearch()
{
	if (lineSearch_type == OptimizationUtils::LineSearch::GRADIENT_NORM)
		gradNorm_linesearch();
	else if (lineSearch_type == OptimizationUtils::LineSearch::FUNCTION_VALUE)
		value_linesearch();
	else if (lineSearch_type == OptimizationUtils::LineSearch::CONSTANT_STEP)
		constant_linesearch();
}

void Basic::value_linesearch()
{	
	step_size = init_step_size;
	int cur_iter = 0; 
	while (cur_iter++ < MAX_STEP_SIZE_ITER) 
	{
		for (int i = 0; i < totalObjective->grad.size; i++) {
			curr_x.host_arr[i] = X.host_arr[i] + step_size * p.host_arr[i];
		}

		double new_energy = totalObjective->value(curr_x,false);
		if (new_energy >= currentEnergy)
			step_size /= 2;
		else 
		{
			for (int i = 0; i < totalObjective->grad.size; i++) {
				X.host_arr[i] = curr_x.host_arr[i];
			}
			break;
		}
	}
	if (cur_iter < MAX_STEP_SIZE_ITER) {
		if (cur_iter == 1)
			init_step_size *= 2;
		if (cur_iter > 2)
			init_step_size /= 2;
	}
	linesearch_numiterations = cur_iter;

	if (isUpdateLambdaWhenConverge) {
		if (linesearch_numiterations >= MAX_STEP_SIZE_ITER) linesearch_StopCounter++;
		else linesearch_StopCounter = 0;
		if (linesearch_StopCounter >= 7) {
			std::shared_ptr<ObjectiveFunctions::Panels::AuxSphere> ASH = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxSphere>(totalObjective->objectiveList[0]);
			std::shared_ptr<ObjectiveFunctions::Panels::AuxPlanar> AP = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxPlanar>(totalObjective->objectiveList[1]);
			std::shared_ptr<ObjectiveFunctions::Panels::Planar> BN = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::Planar>(totalObjective->objectiveList[2]);
			const double target = pow(2, -autoLambda_count);
			ASH->Dec_SigmoidParameter(target);
			AP->Dec_SigmoidParameter(target);
			BN->Dec_SigmoidParameter(target);
		}
	}
}

void Basic::constant_linesearch()
{
	for (int i = 0; i < totalObjective->grad.size; i++) {
		curr_x.host_arr[i] = X.host_arr[i] + constantStep_LineSearch * p.host_arr[i];
		X.host_arr[i] = curr_x.host_arr[i];
	}
}

void Basic::gradNorm_linesearch()
{
}

void Basic::stop()
{
	wait_for_parameter_update_slot();
	halt = true;
	release_parameter_update_slot();
}

void Basic::update_external_data()
{
	give_parameter_update_slot();
	std::unique_lock<std::shared_timed_mutex> lock(*data_mutex);
	for (int i = 0; i < 3 * V.rows(); i++)
		ext_x[i] = X.host_arr[i];
	for (int i = 0; i < 3 * F.rows(); i++)
		ext_norm[i] = X.host_arr[3 * V.rows() + i];
	for (int i = 0; i < 3 * F.rows(); i++)
		ext_center[i] = X.host_arr[3 * V.rows() + 3 * F.rows() + i];
	for (int i = 0; i < F.rows(); i++)
		ext_radius[i] = X.host_arr[3 * V.rows() + 6 * F.rows() + i];
	progressed = true;
}

void Basic::get_data(
	Eigen::MatrixXd& X, 
	Eigen::MatrixXd& center, 
	Eigen::VectorXd& radius, 
	Eigen::MatrixXd& norm)
{
	std::unique_lock<std::shared_timed_mutex> lock(*data_mutex);
	X				= Eigen::Map<Eigen::MatrixXd>(ext_x.data(), ext_x.rows() / 3, 3);
	center			= Eigen::Map<Eigen::MatrixXd>(ext_center.data(), ext_center.rows() / 3, 3);
	radius			= ext_radius;
	norm			= Eigen::Map<Eigen::MatrixXd>(ext_norm.data(), ext_norm.rows() / 3, 3);
	progressed = false;
}

void Basic::give_parameter_update_slot()
{
	std::unique_lock<std::mutex> lock(*parameters_mutex);
	params_ready_to_update = true;
	param_cv->notify_one();
	while (wait_for_param_update)
	{
		param_cv->wait(lock);
	}
	params_ready_to_update = false;
}

void Basic::wait_for_parameter_update_slot()
{
	std::unique_lock<std::mutex> lock(*parameters_mutex);
	wait_for_param_update = true;
	while (!params_ready_to_update && is_running)
		param_cv->wait_for(lock, std::chrono::milliseconds(50));
}

void Basic::release_parameter_update_slot()
{
	wait_for_param_update = false;
	param_cv->notify_one();
}