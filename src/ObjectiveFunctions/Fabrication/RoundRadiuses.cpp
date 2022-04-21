#include "ObjectiveFunctions/Fabrication/RoundRadiuses.h"

fixRadius::fixRadius(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) : ObjectiveFunctions::Basic{ V,F }
{
	name = "fix Radius";
	w = 0;
	std::cout << "\t" << name << " constructor" << std::endl;
}

fixRadius::~fixRadius() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

double fixRadius::value(Cuda::Array<double>& curr_x, const bool update) {
	double value = 0;
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		double R = getR(curr_x, fi);
		double AR = alpha * R;

		if (AR < 1.25) {
			value += pow(AR - 1.0, 2);
		}
		else if (1.25 <= AR && AR < 1.75) {
			value += pow(AR - 1.5, 2);
		}
		else if (1.75 <= AR && AR < 2.25) {
			value += pow(AR - 2, 2);
		}
		else if (2.25 <= AR && AR < 2.75) {
			value += pow(AR - 2.5, 2);
		}
		else if (2.75 <= AR && AR < 3.25) {
			value += pow(AR - 3, 2);
		}
		else if (3.25 <= AR && AR < 3.75) {
			value += pow(AR - 3.5, 2);
		}
		else if (3.75 <= AR && AR < 4.5) {
			value += pow(AR - 4, 2);
		}
		else if (4.5 <= AR && AR < 5.5) {
			value += pow(AR - 5, 2);
		}
		else if (5.5 <= AR && AR < 6.5) {
			value += pow(AR - 6, 2);
		}
		else if (6.5 <= AR) {
			value += pow(AR - 7, 2);
		}
	}
	if (update)
		energy_value = value;
	return value;
}

void fixRadius::gradient(Cuda::Array<double>& X,const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const int startR = mesh_indices.startR;
		double R = getR(X, fi);
		double AR = alpha * R;

		if (AR < 1.25) {
			grad.host_arr[fi + startR] += 2 * alpha * (AR - 1.0);
		}
		else if (1.25 <= AR && AR < 1.75) {
			grad.host_arr[fi + startR] += 2 * alpha * (AR - 1.5);
		}
		else if (1.75 <= AR && AR < 2.25) {
			grad.host_arr[fi + startR] += 2 * alpha * (AR - 2.0);
		}
		else if (2.25 <= AR && AR < 2.75) {
			grad.host_arr[fi + startR] += 2 * alpha * (AR - 2.5);
		}
		else if (2.75 <= AR && AR < 3.25) {
			grad.host_arr[fi + startR] += 2 * alpha * (AR - 3.0);
		}
		else if (3.25 <= AR && AR < 3.75) {
			grad.host_arr[fi + startR] += 2 * alpha * (AR - 3.5);
		}
		else if (3.75 <= AR && AR < 4.5) {
			grad.host_arr[fi + startR] += 2 * alpha * (AR - 4.0);
		}
		else if (4.5 <= AR && AR < 5.5) {
			grad.host_arr[fi + startR] += 2 * alpha * (AR - 5.0);
		}
		else if (5.5 <= AR && AR < 6.5) {
			grad.host_arr[fi + startR] += 2 * alpha * (AR - 6.0);
		}
		else if (6.5 <= AR) {
			grad.host_arr[fi + startR] += 2 * alpha * (AR - 7.0);
		}
	}

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}
