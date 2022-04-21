#include "AuxBendingNormal.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>

AuxBendingNormal::AuxBendingNormal(
	const Eigen::MatrixXd& V, 
	const Eigen::MatrixX3i& F,
	const Cuda::PenaltyFunction penaltyFunction) : AuxVariables{V,F,penaltyFunction}
{
	name = "Aux Bending Normal";
	std::cout << "\t" << name << " constructor" << std::endl;
}

AuxBendingNormal::~AuxBendingNormal() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

double AuxBendingNormal::value(Cuda::Array<double>& curr_x, const bool update)
{
	double value = 0;
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi][0];
		int f1 = hinges_faceIndex[hi][1];
		double_3 N0 = getN(curr_x, f0);
		double_3 N1 = getN(curr_x, f1);
		double_3 diff = sub(N1, N0);
		double d_normals = squared_norm(diff);
		value += w1 * restAreaPerHinge[hi] * weight_PerHinge.host_arr[hi] *
			Phi(d_normals, Sigmoid_PerHinge.host_arr[hi], penaltyFunction);
	}

	for (int fi = 0; fi < mesh_indices.num_faces; fi++) {
		// (N^T*(x1-x0))^2 + (N^T*(x2-x1))^2 + (N^T*(x0-x2))^2
		const int x0 = restShapeF(fi, 0);
		const int x1 = restShapeF(fi, 1);
		const int x2 = restShapeF(fi, 2);
		double_3 V0 = getV(curr_x, x0);
		double_3 V1 = getV(curr_x, x1);
		double_3 V2 = getV(curr_x, x2);
		double_3 N = getN(curr_x, fi);
		
		double_3 e21 = sub(V2, V1);
		double_3 e10 = sub(V1, V0);
		double_3 e02 = sub(V0, V2);
		value += w3 * (pow(dot(N, e21), 2) + pow(dot(N, e10), 2) + pow(dot(N, e02), 2));
		value += pow(squared_norm(N) - 1, 2) * w2;
	}
	
	if (update)
		energy_value = value;
	return value;
}

void AuxBendingNormal::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++) {
		grad.host_arr[i] = 0;
	}

	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi][0];
		int f1 = hinges_faceIndex[hi][1];
		double_3 N0 = getN(X, f0); 
		double_3 N1 = getN(X, f1); 
		double_3 diff = sub(N1, N0);
		double d_normals = squared_norm(diff);

		double coeff = 2 * w1 * restAreaPerHinge[hi] * weight_PerHinge.host_arr[hi] *
			dPhi_dm(d_normals, Sigmoid_PerHinge.host_arr[hi], penaltyFunction);

		grad.host_arr[f0 + mesh_indices.startNx] += coeff * (N0.x - N1.x);
		grad.host_arr[f1 + mesh_indices.startNx] += coeff * (N1.x - N0.x);
		grad.host_arr[f0 + mesh_indices.startNy] += coeff * (N0.y - N1.y);
		grad.host_arr[f1 + mesh_indices.startNy] += coeff * (N1.y - N0.y);
		grad.host_arr[f0 + mesh_indices.startNz] += coeff * (N0.z - N1.z);
		grad.host_arr[f1 + mesh_indices.startNz] += coeff * (N1.z - N0.z);
	}
	

	for (int fi = 0; fi < mesh_indices.num_faces; fi++) {
		const unsigned int x0 = restShapeF(fi, 0);
		const unsigned int x1 = restShapeF(fi, 1);
		const unsigned int x2 = restShapeF(fi, 2);
		double_3 V0 = getV(X, x0); 
		double_3 V1 = getV(X, x1); 
		double_3 V2 = getV(X, x2); 
		double_3 N = getN(X, fi); 

		double_3 e21 = sub(V2, V1);
		double_3 e10 = sub(V1, V0);
		double_3 e02 = sub(V0, V2);
		double N02 = dot(N, e02);
		double N10 = dot(N, e10);
		double N21 = dot(N, e21);
		double coeff = 2 * w3;
		double coeff2 = w2 * 4 * (squared_norm(N) - 1);

		grad.host_arr[x0 + mesh_indices.startVx] += coeff * N.x * (N02 - N10);
		grad.host_arr[x0 + mesh_indices.startVy] += coeff * N.y * (N02 - N10);
		grad.host_arr[x0 + mesh_indices.startVz] += coeff * N.z * (N02 - N10);
		grad.host_arr[x1 + mesh_indices.startVx] += coeff * N.x * (N10 - N21);
		grad.host_arr[x1 + mesh_indices.startVy] += coeff * N.y * (N10 - N21);
		grad.host_arr[x1 + mesh_indices.startVz] += coeff * N.z * (N10 - N21);
		grad.host_arr[x2 + mesh_indices.startVx] += coeff * N.x * (N21 - N02);
		grad.host_arr[x2 + mesh_indices.startVy] += coeff * N.y * (N21 - N02);
		grad.host_arr[x2 + mesh_indices.startVz] += coeff * N.z * (N21 - N02);
		grad.host_arr[fi + mesh_indices.startNx] += (coeff2 * N.x) + (coeff * (N10 * e10.x + N21 * e21.x + N02 * e02.x));
		grad.host_arr[fi + mesh_indices.startNy] += (coeff2 * N.y) + (coeff * (N10 * e10.y + N21 * e21.y + N02 * e02.y));
		grad.host_arr[fi + mesh_indices.startNz] += (coeff2 * N.z) + (coeff * (N10 * e10.z + N21 * e21.z + N02 * e02.z));
	}
	
	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}

