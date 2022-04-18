#include "UniformSmoothness.h"
#include <igl/adjacency_matrix.h>
#include <igl/sum.h>
#include <igl/diag.h>

UniformSmoothness::UniformSmoothness(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F)
	: ObjectiveFunction{ V,F }
{
	name = "Uniform Smoothness";
	w = 0.05;
	
	// Prepare The Uniform Laplacian 
	// Mesh in (V,F)
	Eigen::SparseMatrix<double> A;
	igl::adjacency_matrix((Eigen::MatrixXi)F, A);
    // sum each row 
	Eigen::SparseVector<double> Asum;
	igl::sum(A,1,Asum);
    // Convert row sums into diagonal of sparse matrix
	Eigen::SparseMatrix<double> Adiag;
    igl::diag(Asum,Adiag);
    // Build uniform laplacian
    L = A-Adiag;

	L2 = 2 * L * L;

	std::cout << "\t" << name << " constructor" << std::endl;
}

UniformSmoothness::~UniformSmoothness() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

double UniformSmoothness::value(Cuda::Array<double>& curr_x, const bool update) {
	// Energy = ||L * x||^2
	// Energy = ||diag(L,L,L) * (x;y;z)||^2
	double value = 0;
	Eigen::VectorXd X(restShapeV.rows()), Y(restShapeV.rows()), Z(restShapeV.rows());
	for (int vi = 0; vi < restShapeV.rows(); vi++) {
		X(vi) = curr_x.host_arr[vi + mesh_indices.startVx];
		Y(vi) = curr_x.host_arr[vi + mesh_indices.startVy];
		Z(vi) = curr_x.host_arr[vi + mesh_indices.startVz];
	}
	value =
		(L * X).squaredNorm() + 
		(L * Y).squaredNorm() + 
		(L * Z).squaredNorm();
	if (update)
		energy_value = value;
	return value;
}

void UniformSmoothness::gradient(Cuda::Array<double>& input_X, const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;
	// gradient = 2*||L * x|| * L
	Eigen::VectorXd X(restShapeV.rows()), Y(restShapeV.rows()), Z(restShapeV.rows());
	for (int vi = 0; vi < restShapeV.rows(); vi++) {
		X(vi) = input_X.host_arr[vi + mesh_indices.startVx];
		Y(vi) = input_X.host_arr[vi + mesh_indices.startVy];
		Z(vi) = input_X.host_arr[vi + mesh_indices.startVz];
	}
	Eigen::VectorXd grad_X = L2 * X;
	Eigen::VectorXd grad_Y = L2 * Y;
	Eigen::VectorXd grad_Z = L2 * Z;


	for (int vi = 0; vi < restShapeV.rows(); vi++) {
		grad.host_arr[vi + mesh_indices.startVx] += grad_X(vi);
		grad.host_arr[vi + mesh_indices.startVy] += grad_Y(vi);
		grad.host_arr[vi + mesh_indices.startVz] += grad_Z(vi);
	}

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}
