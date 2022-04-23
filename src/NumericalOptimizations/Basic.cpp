#include "NumericalOptimizations/Basic.h"
#include "ObjectiveFunctions/Panels/AuxPlanar.h"
#include "ObjectiveFunctions/Panels/AuxCylinder1.h"
#include "ObjectiveFunctions/Panels/AuxCylinder2.h"
#include "ObjectiveFunctions/Panels/AuxCylinder3.h"
#include "ObjectiveFunctions/Panels/Planar.h"
#include "ObjectiveFunctions/Panels/AuxSphere.h"
#include "ObjectiveFunctions/Deformation/SymmetricDirichlet.h"

#define BETA1_ADAM 0.90f
#define BETA2_ADAM 0.9990f
#define EPSILON_ADAM 1e-8
#define MAX_STEP_SIZE_ITER 50

using namespace NumericalOptimizations;


Basic::Basic(const int solverID){
	this->solverID = solverID;
	this->constantStep_LineSearch = 0.01;
}

void Basic::init(
	std::shared_ptr<ObjectiveFunctions::Total> Tobjective,
	const Eigen::MatrixXd& V0,
	const Eigen::MatrixXi& F0,
	const Eigen::MatrixXd& norm0,
	const Eigen::MatrixXd& center0,
	const Eigen::MatrixXd& radius0,
	const Eigen::MatrixXd& cylinder_dir0)
{
	assert(Tobjective != NULL);
	assert(V0.rows() >= 3 && V0.cols() == 3);
	assert(F0.rows() >= 1 && F0.cols() == 3);
	assert(norm0.rows() == F0.rows() && norm0.cols() == 3);
	assert(center0.rows() == F0.rows() && center0.cols() == 3);
	assert(radius0.rows() == F0.rows() && radius0.cols() == 1);
	assert(cylinder_dir0.rows() == F0.rows() && cylinder_dir0.cols() == 3);
	
	
	this->totalObjective = Tobjective;

	int numH = OptimizationUtils::getNumberOfHinges(F0);
	Cuda::initIndices(mesh_indices, F0.rows(), V0.rows(), numH);

	Cuda::AllocateMemory(X, mesh_indices.total_variables);
	Cuda::AllocateMemory(p, mesh_indices.total_variables);
	Cuda::AllocateMemory(curr_x, mesh_indices.total_variables);
	Cuda::AllocateMemory(v_adam, mesh_indices.total_variables);
	Cuda::AllocateMemory(s_adam, mesh_indices.total_variables);
	
	for (int vi = 0; vi < mesh_indices.num_vertices; vi++) {
		X.host_arr[vi + mesh_indices.startVx] = V0(vi, 0);
		X.host_arr[vi + mesh_indices.startVy] = V0(vi, 1);
		X.host_arr[vi + mesh_indices.startVz] = V0(vi, 2);
	}
	for (int fi = 0; fi < mesh_indices.num_faces; fi++) {
		X.host_arr[fi + mesh_indices.startNx] = norm0(fi, 0);
		X.host_arr[fi + mesh_indices.startNy] = norm0(fi, 1);
		X.host_arr[fi + mesh_indices.startNz] = norm0(fi, 2);
		X.host_arr[fi + mesh_indices.startCx] = center0(fi, 0);
		X.host_arr[fi + mesh_indices.startCy] = center0(fi, 1);
		X.host_arr[fi + mesh_indices.startCz] = center0(fi, 2);
		X.host_arr[fi + mesh_indices.startR] = radius0(fi);
		X.host_arr[fi + mesh_indices.startAx] = cylinder_dir0(fi, 0);
		X.host_arr[fi + mesh_indices.startAy] = cylinder_dir0(fi, 1);
		X.host_arr[fi + mesh_indices.startAz] = cylinder_dir0(fi, 2);
	}
	for (int i = 0; i < mesh_indices.total_variables; i++) {
		v_adam.host_arr[i] = 0;
		s_adam.host_arr[i] = 0;
		p.host_arr[i] = 0;
		curr_x.host_arr[i] = X.host_arr[i];
	}
}

void Basic::upload_x(const Eigen::VectorXd& X0)
{
	assert(X0.rows() == (3 * mesh_indices.num_vertices));
	for (int i = 0; i < 3 * mesh_indices.num_vertices; i++)
		X.host_arr[i] = X0[i];
	for (int i = 0; i < mesh_indices.total_variables; i++)
		curr_x.host_arr[i] = X.host_arr[i];
}

void Basic::run()
{
	is_running = true;
	halt = false;
	while (!halt)
		run_one_iteration();
	std::cout << ">> solver " + std::to_string(solverID) + " stopped" << std::endl;
	is_running = false;
}

void Basic::run_new()
{
	is_running = true;
	halt = false;
	while (!halt)
		run_one_iteration_new();
	std::cout << ">> solver " + std::to_string(solverID) + " stopped" << std::endl;
	is_running = false;
}

void Basic::RunSymmetricDirichletGradient() {
	halt = false;
	while (!halt) {
		std::shared_ptr<ObjectiveFunctions::Deformation::SymmetricDirichlet> SD = std::dynamic_pointer_cast<ObjectiveFunctions::Deformation::SymmetricDirichlet>(totalObjective->objectiveList[7]);
		if (isGradientNeeded) {
			if (SD->w != 0)
				SD->gradient(X, false);
			isGradientNeeded = false;
		}
	}
}

void Basic::update_lambda()
{
	std::shared_ptr<ObjectiveFunctions::Panels::AuxCylinder0> AC1 =
		std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxCylinder0>(totalObjective->objectiveList[0]);
	std::shared_ptr<ObjectiveFunctions::Panels::AuxCylinder2> AC2 =
		std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxCylinder2>(totalObjective->objectiveList[1]);
	std::shared_ptr<ObjectiveFunctions::Panels::AuxCylinder3> AC3 =
		std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxCylinder3>(totalObjective->objectiveList[2]);
	std::shared_ptr<ObjectiveFunctions::Panels::AuxSphere> ASH = 
		std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxSphere>(totalObjective->objectiveList[3]);
	std::shared_ptr<ObjectiveFunctions::Panels::AuxPlanar> ABN = 
		std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxPlanar>(totalObjective->objectiveList[4]);
	std::shared_ptr<ObjectiveFunctions::Panels::Planar> BN =
		std::dynamic_pointer_cast<ObjectiveFunctions::Panels::Planar>(totalObjective->objectiveList[5]);
	
	if (isAutoLambdaRunning && numIteration >= autoLambda_from && !(numIteration % autoLambda_jump))
	{
		const double target = pow(2, -autoLambda_count);
		AC1->Dec_SigmoidParameter(target);
		AC2->Dec_SigmoidParameter(target);
		AC3->Dec_SigmoidParameter(target);
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
		for (int i = 0; i < mesh_indices.total_variables; i++) {
			v_adam.host_arr[i] = BETA1_ADAM * v_adam.host_arr[i] + (1 - BETA1_ADAM) * totalObjective->grad.host_arr[i];
			s_adam.host_arr[i] = BETA2_ADAM * s_adam.host_arr[i] + (1 - BETA2_ADAM) * pow(totalObjective->grad.host_arr[i], 2);
			p.host_arr[i] = -v_adam.host_arr[i] / (sqrt(s_adam.host_arr[i]) + EPSILON_ADAM);
		}
	}
	else if (Optimizer_type == Cuda::OptimizerType::Gradient_Descent) {
		for (int i = 0; i < mesh_indices.total_variables; i++) {
			p.host_arr[i] = -totalObjective->grad.host_arr[i];
		}
	}
	currentEnergy = totalObjective->value(X, true);
	linesearch();
}

void Basic::run_one_iteration_new()
{
	OptimizationUtils::Timer t(&timer_sum, &timer_curr);
	//calculate SD gradient in advance
	isGradientNeeded = true;
	std::shared_ptr<ObjectiveFunctions::Deformation::SymmetricDirichlet> SD = std::dynamic_pointer_cast<ObjectiveFunctions::Deformation::SymmetricDirichlet>(totalObjective->objectiveList[7]);
	
	timer_avg = timer_sum / ++numIteration;
	update_lambda();

	//////////////////////////
	//Get the first part of the gradients
	for (int i = 0; i < mesh_indices.total_variables; i++)
		totalObjective->grad.host_arr[i] = 0;
	for (auto& obj : totalObjective->objectiveList) {
		std::shared_ptr<ObjectiveFunctions::Deformation::SymmetricDirichlet> SD = std::dynamic_pointer_cast<ObjectiveFunctions::Deformation::SymmetricDirichlet>(obj);
		if (obj->w != 0 && SD == NULL) {
			obj->gradient(X, false);
			for (int i = 0; i < mesh_indices.total_variables; i++)
				totalObjective->grad.host_arr[i] += obj->w * obj->grad.host_arr[i];
		}
	}
	

	currentEnergy = totalObjective->value(X, true);

	//////////////////////////
	///get the second part of the gradient
	while (isGradientNeeded);
	if (SD->w != 0) {
		for (int i = 0; i < mesh_indices.total_variables; i++)
			totalObjective->grad.host_arr[i] += SD->w * SD->grad.host_arr[i];
	}
	//////////////////////////

	if (Optimizer_type == Cuda::OptimizerType::Adam) {
		for (int i = 0; i < mesh_indices.total_variables; i++) {
			v_adam.host_arr[i] = BETA1_ADAM * v_adam.host_arr[i] + (1 - BETA1_ADAM) * totalObjective->grad.host_arr[i];
			s_adam.host_arr[i] = BETA2_ADAM * s_adam.host_arr[i] + (1 - BETA2_ADAM) * pow(totalObjective->grad.host_arr[i], 2);
			p.host_arr[i] = -v_adam.host_arr[i] / (sqrt(s_adam.host_arr[i]) + EPSILON_ADAM);
		}
	}
	else if (Optimizer_type == Cuda::OptimizerType::Gradient_Descent) {
		for (int i = 0; i < mesh_indices.total_variables; i++) {
			p.host_arr[i] = -totalObjective->grad.host_arr[i];
		}
	}
	linesearch();
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
		for (int i = 0; i < mesh_indices.total_variables; i++) {
			curr_x.host_arr[i] = X.host_arr[i] + step_size * p.host_arr[i];
		}

		double new_energy = totalObjective->value(curr_x,false);
		if (new_energy >= currentEnergy)
			step_size /= 2;
		else 
		{
			for (int i = 0; i < mesh_indices.total_variables; i++) {
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
			std::shared_ptr<ObjectiveFunctions::Panels::AuxCylinder0> AC1 = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxCylinder0>(totalObjective->objectiveList[0]);
			std::shared_ptr<ObjectiveFunctions::Panels::AuxCylinder2> AC2 = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxCylinder2>(totalObjective->objectiveList[1]);
			std::shared_ptr<ObjectiveFunctions::Panels::AuxCylinder3> AC3 = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxCylinder3>(totalObjective->objectiveList[2]);
			std::shared_ptr<ObjectiveFunctions::Panels::AuxSphere> ASH = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxSphere>(totalObjective->objectiveList[3]);
			std::shared_ptr<ObjectiveFunctions::Panels::AuxPlanar> AP = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxPlanar>(totalObjective->objectiveList[4]);
			std::shared_ptr<ObjectiveFunctions::Panels::Planar> BN = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::Planar>(totalObjective->objectiveList[5]);
			const double target = pow(2, -autoLambda_count);
			AC1->Dec_SigmoidParameter(target);
			AC2->Dec_SigmoidParameter(target);
			AC3->Dec_SigmoidParameter(target);
			ASH->Dec_SigmoidParameter(target);
			AP->Dec_SigmoidParameter(target);
			BN->Dec_SigmoidParameter(target);
		}
	}
}

void Basic::constant_linesearch()
{
	for (int i = 0; i < mesh_indices.total_variables; i++) {
		curr_x.host_arr[i] = X.host_arr[i] + constantStep_LineSearch * p.host_arr[i];
		X.host_arr[i] = curr_x.host_arr[i];
	}
}

void Basic::gradNorm_linesearch()
{
}

bool Basic::external_is_running()
{
	return is_running;
}

void Basic::external_stop()
{
	halt = true;
	while (is_running != false);
}

void Basic::external_get_data(
	Eigen::MatrixXd& X, 
	Eigen::MatrixXd& center, 
	Eigen::VectorXd& radius, 
	Eigen::MatrixXd& norm,
	Eigen::MatrixXd& cylinder_dir)
{
	for (int vi = 0; vi < mesh_indices.num_vertices; vi++) {
		X(vi, 0) = this->X.host_arr[vi + mesh_indices.startVx];
		X(vi, 1) = this->X.host_arr[vi + mesh_indices.startVy];
		X(vi, 2) = this->X.host_arr[vi + mesh_indices.startVz];
	}
	for (int fi = 0; fi < mesh_indices.num_faces; fi++) {
		center(fi, 0) = this->X.host_arr[fi + mesh_indices.startCx];
		center(fi, 1) = this->X.host_arr[fi + mesh_indices.startCy];
		center(fi, 2) = this->X.host_arr[fi + mesh_indices.startCz];
		cylinder_dir(fi, 0) = this->X.host_arr[fi + mesh_indices.startAx];
		cylinder_dir(fi, 1) = this->X.host_arr[fi + mesh_indices.startAy];
		cylinder_dir(fi, 2) = this->X.host_arr[fi + mesh_indices.startAz];
		norm(fi, 0) = this->X.host_arr[fi + mesh_indices.startNx];
		norm(fi, 1) = this->X.host_arr[fi + mesh_indices.startNy];
		norm(fi, 2) = this->X.host_arr[fi + mesh_indices.startNz];
		radius(fi) = this->X.host_arr[fi + mesh_indices.startR];
	}
}