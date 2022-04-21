#include "ObjectiveFunctions/Total.h"

using namespace ObjectiveFunctions;

Total::Total(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) : ObjectiveFunctions::Basic{ V,F }
{
	name = "Total";
	std::cout << "\t" << name << " constructor" << std::endl;
}

Total::~Total()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

double Total::value(Cuda::Array<double>& curr_x, const bool update)
{
	double value = 0;
	for (auto& obj : objectiveList)
		if (obj->w != 0)
			value += obj->w * obj->value(curr_x, update);
	
	if (update)
		energy_value = value;
	return value;
}

double Total::value_print(Cuda::Array<double>& curr_x, const bool update)
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

void Total::gradient(Cuda::Array<double>& X, const bool update)
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