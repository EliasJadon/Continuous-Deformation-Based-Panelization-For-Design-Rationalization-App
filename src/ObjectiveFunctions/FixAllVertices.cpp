#include "FixAllVertices.h"

FixAllVertices::FixAllVertices(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixX3i& F) : ObjectiveFunction{ V,F }
{
	name = "Fix All Vertices";
	w = 0.3;
	std::cout << "\t" << name << " constructor" << std::endl;
}

FixAllVertices::~FixAllVertices()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

double FixAllVertices::value(Cuda::Array<double>& curr_x, const bool update)
{
	double value = 0;
	for (int vi = 0; vi < restShapeV.rows(); vi++) {
		double_3 V = getV(curr_x, vi);
		value +=
			pow(V.x - restShapeV(vi, 0), 2) +
			pow(V.y - restShapeV(vi, 1), 2) +
			pow(V.z - restShapeV(vi, 2), 2);
	}
	if (update)
		energy_value = value;
	return value;
}

void FixAllVertices::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;

	for (int vi = 0; vi < restShapeV.rows(); vi++) {
		double_3 V = getV(X, vi);
		grad.host_arr[vi + mesh_indices.startVx] += 2 * (V.x - restShapeV(vi, 0));
		grad.host_arr[vi + mesh_indices.startVy] += 2 * (V.y - restShapeV(vi, 1));
		grad.host_arr[vi + mesh_indices.startVz] += 2 * (V.z - restShapeV(vi, 2));
	}

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}
