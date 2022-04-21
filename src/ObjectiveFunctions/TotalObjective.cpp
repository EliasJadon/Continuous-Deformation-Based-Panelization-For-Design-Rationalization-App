#include "TotalObjective.h"

TotalObjective::TotalObjective(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) : ObjectiveFunction{ V,F }
{
	name = "Total objective";
	std::cout << "\t" << name << " constructor" << std::endl;
}

TotalObjective::~TotalObjective()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

double TotalObjective::value(Cuda::Array<double>& curr_x, const bool update)
{
	double value = 0;
	for (auto& obj : objectiveList)
		if (obj->w != 0)
			value += obj->w * obj->value(curr_x, update);
	
	if (update)
		energy_value = value;
	return value;
}

double TotalObjective::value_print(Cuda::Array<double>& curr_x, const bool update)
{
	double value = 0;
	for (auto& obj : objectiveList)
		if (obj->w != 0) {
			double val = obj->w * obj->value(curr_x, update);
			value += val;
			std::cout << val << ", ";
		}
	std::cout << value << ",\n";
	if (update)
		energy_value = value;
	return value;
}

void TotalObjective::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;

	for (auto& obj : objectiveList) {
		if (obj->w != 0) {
			obj->gradient(X, update);
			for (int i = 0; i < grad.size; i++)
				grad.host_arr[i] += obj->w * obj->grad.host_arr[i];
		}
	}
		
	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}