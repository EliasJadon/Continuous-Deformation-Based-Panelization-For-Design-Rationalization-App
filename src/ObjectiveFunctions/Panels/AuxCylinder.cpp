#include "ObjectiveFunctions/Panels/AuxCylinder.h"

using namespace ObjectiveFunctions::Panels;

AuxCylinder::AuxCylinder(
	const Eigen::MatrixXd& V, 
	const Eigen::MatrixX3i& F,
	const Cuda::PenaltyFunction type) : ObjectiveFunctions::Panels::AuxBasic{V,F,type}
{
	name = "Aux Cylinder";
	std::cout << "\t" << name << " constructor" << std::endl;
}

AuxCylinder::~AuxCylinder() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

double AuxCylinder::value(Cuda::Array<double>& curr_x, const bool update)
{	
	double value = 0;
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi][0];
		int f1 = hinges_faceIndex[hi][1];
		double_3 C0 = getC(curr_x, f0);
		double_3 A0 = getA(curr_x, f0);
		double R0 = getR(curr_x, f0);
		double_3 C1 = getC(curr_x, f1);
		double_3 A1 = getA(curr_x, f1);
		double R1 = getR(curr_x, f1);
		double_3 C10 = sub(C1, C0);

		double diff =
			pow(R1 - R0, 2) +
			pow(pow(dot(A1, A0), 2) - 1, 2) +
			pow(pow(dot(C10, A0), 2) - squared_norm(C10), 2);

		/*value += w1 * restAreaPerHinge[hi] * weight_PerHinge.host_arr[hi] *
			Phi(diff, Sigmoid_PerHinge.host_arr[hi], penaltyFunction);*/
	}

	for (int fi = 0; fi < mesh_indices.num_faces; fi++) {
		const int x0 = restShapeF(fi, 0);
		const int x1 = restShapeF(fi, 1);
		const int x2 = restShapeF(fi, 2);
		double_3 V0 = getV(curr_x, x0);
		double_3 V1 = getV(curr_x, x1);
		double_3 V2 = getV(curr_x, x2);
		double_3 C = getC(curr_x, fi);
		double_3 A = getA(curr_x, fi);
		double R = getR(curr_x, fi);
		double R2 = pow(R, 2);

		value += w2 * pow(squared_norm(A) - 1, 2);
		/*value += w3 * (
			pow(squared_norm(cross(A, sub(V0, C))) - R2, 2) +
			pow(squared_norm(cross(A, sub(V1, C))) - R2, 2) +
			pow(squared_norm(cross(A, sub(V2, C))) - R2, 2)
			);*/
	}

	if (update)
		energy_value = value;
	return value;
}

void AuxCylinder::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;

	for (int hi = 0; hi < num_hinges; hi++) {
		
	}
	
	for (int fi = 0; fi < mesh_indices.num_faces; fi++) {
		const int x0 = restShapeF(fi, 0);
		const int x1 = restShapeF(fi, 1);
		const int x2 = restShapeF(fi, 2);
		double_3 V0 = getV(X, x0);
		double_3 V1 = getV(X, x1);
		double_3 V2 = getV(X, x2);
		double_3 C = getC(X, fi);
		double_3 A = getA(X, fi);
		double R = getR(X, fi);
		double R2 = pow(R, 2);

		double coeff_E2 = w2 * 4 * (squared_norm(A) - 1);

		double coeff_E3_V0 = w3 * 2 * (squared_norm(cross(A, sub(V0, C))) - R2);
		double coeff_E3_V1 = w3 * 2 * (squared_norm(cross(A, sub(V1, C))) - R2);
		double coeff_E3_V2 = w3 * 2 * (squared_norm(cross(A, sub(V2, C))) - R2);
		
		double_3 len_V0 = cross(A, sub(V0, C));
		double_3 len_V1 = cross(A, sub(V1, C));
		double_3 len_V2 = cross(A, sub(V2, C));

		double coeff_E3_squared_norm_V0_X = 2 * len_V0.x;
		double coeff_E3_squared_norm_V0_Y = 2 * len_V0.y;
		double coeff_E3_squared_norm_V0_Z = 2 * len_V0.z;

		double coeff_E3_squared_norm_V1_X = 2 * len_V1.x;
		double coeff_E3_squared_norm_V1_Y = 2 * len_V1.y;
		double coeff_E3_squared_norm_V1_Z = 2 * len_V1.z;

		double coeff_E3_squared_norm_V2_X = 2 * len_V2.x;
		double coeff_E3_squared_norm_V2_Y = 2 * len_V2.y;
		double coeff_E3_squared_norm_V2_Z = 2 * len_V2.z;
		
		grad.host_arr[fi + mesh_indices.startAx] += coeff_E2 * A.x; // Ax
		grad.host_arr[fi + mesh_indices.startAy] += coeff_E2 * A.y; // Ay
		grad.host_arr[fi + mesh_indices.startAz] += coeff_E2 * A.z; // Az
		//grad.host_arr[fi + mesh_indices.startR] += -2 * R * (coeff_E3_V0 + coeff_E3_V1 + coeff_E3_V2); // R

		//grad.host_arr[fi + mesh_indices.startAx] += (V0.y - C.y)* coeff_E3_squared_norm_V0_Z * coeff_E3_V0; //Ax
		//grad.host_arr[fi + mesh_indices.startAx] += -(V0.z - C.z) * coeff_E3_squared_norm_V0_Y * coeff_E3_V0;
		//grad.host_arr[fi + mesh_indices.startAx] += (V1.y - C.y)* coeff_E3_squared_norm_V1_Z* coeff_E3_V1; //Ax
		//grad.host_arr[fi + mesh_indices.startAx] += -(V1.z - C.z) * coeff_E3_squared_norm_V1_Y * coeff_E3_V1;
		//grad.host_arr[fi + mesh_indices.startAx] += (V2.y - C.y)* coeff_E3_squared_norm_V2_Z* coeff_E3_V2; //Ax
		//grad.host_arr[fi + mesh_indices.startAx] += -(V2.z - C.z) * coeff_E3_squared_norm_V2_Y * coeff_E3_V2;

		//grad.host_arr[fi + mesh_indices.startAy] += (V0.z - C.z)* coeff_E3_squared_norm_V0_X* coeff_E3_V0; //Ay
		//grad.host_arr[fi + mesh_indices.startAy] += -(V0.x - C.x) * coeff_E3_squared_norm_V0_Z * coeff_E3_V0;
		//grad.host_arr[fi + mesh_indices.startAy] += (V1.z - C.z)* coeff_E3_squared_norm_V1_X* coeff_E3_V1; //Ay
		//grad.host_arr[fi + mesh_indices.startAy] += -(V1.x - C.x) * coeff_E3_squared_norm_V1_Z * coeff_E3_V1;
		//grad.host_arr[fi + mesh_indices.startAy] += (V2.z - C.z)* coeff_E3_squared_norm_V2_X* coeff_E3_V2; //Ay
		//grad.host_arr[fi + mesh_indices.startAy] += -(V2.x - C.x) * coeff_E3_squared_norm_V2_Z * coeff_E3_V2;

		//grad.host_arr[fi + mesh_indices.startAz] += (V0.x - C.x)* coeff_E3_squared_norm_V0_Y* coeff_E3_V0; //Az
		//grad.host_arr[fi + mesh_indices.startAz] += -(V0.y - C.y) * coeff_E3_squared_norm_V0_X * coeff_E3_V0;
		//grad.host_arr[fi + mesh_indices.startAz] += (V1.x - C.x)* coeff_E3_squared_norm_V1_Y* coeff_E3_V1; //Az
		//grad.host_arr[fi + mesh_indices.startAz] += -(V1.y - C.y) * coeff_E3_squared_norm_V1_X * coeff_E3_V1;
		//grad.host_arr[fi + mesh_indices.startAz] += (V2.x - C.x)* coeff_E3_squared_norm_V2_Y* coeff_E3_V2; //Az
		//grad.host_arr[fi + mesh_indices.startAz] += -(V2.y - C.y) * coeff_E3_squared_norm_V2_X * coeff_E3_V2;

		//grad.host_arr[x0 + mesh_indices.startVx] += A.z * coeff_E3_squared_norm_V0_Y * coeff_E3_V0; // V0.x
		//grad.host_arr[x0 + mesh_indices.startVx] += -A.y * coeff_E3_squared_norm_V0_Z * coeff_E3_V0;

		//grad.host_arr[x0 + mesh_indices.startVy] += A.x* coeff_E3_squared_norm_V0_Z* coeff_E3_V0; // V0.y
		//grad.host_arr[x0 + mesh_indices.startVy] += -A.z * coeff_E3_squared_norm_V0_X * coeff_E3_V0;

		//grad.host_arr[x0 + mesh_indices.startVz] += A.y* coeff_E3_squared_norm_V0_X* coeff_E3_V0; // V0.z
		//grad.host_arr[x0 + mesh_indices.startVz] += -A.x * coeff_E3_squared_norm_V0_Y * coeff_E3_V0;

		//grad.host_arr[x1 + mesh_indices.startVx] += A.z* coeff_E3_squared_norm_V1_Y* coeff_E3_V1; // V1.x
		//grad.host_arr[x1 + mesh_indices.startVx] += -A.y * coeff_E3_squared_norm_V1_Z * coeff_E3_V1;

		//grad.host_arr[x1 + mesh_indices.startVy] += A.x* coeff_E3_squared_norm_V1_Z* coeff_E3_V1; // V1.y
		//grad.host_arr[x1 + mesh_indices.startVy] += -A.z * coeff_E3_squared_norm_V1_X * coeff_E3_V1;

		//grad.host_arr[x1 + mesh_indices.startVz] += A.y* coeff_E3_squared_norm_V1_X* coeff_E3_V1; // V1.z
		//grad.host_arr[x1 + mesh_indices.startVz] += -A.x * coeff_E3_squared_norm_V1_Y * coeff_E3_V1;

		//grad.host_arr[x2 + mesh_indices.startVx] += A.z* coeff_E3_squared_norm_V2_Y* coeff_E3_V2; // V2.x
		//grad.host_arr[x2 + mesh_indices.startVx] += -A.y * coeff_E3_squared_norm_V2_Z * coeff_E3_V2;

		//grad.host_arr[x2 + mesh_indices.startVy] += A.x* coeff_E3_squared_norm_V2_Z* coeff_E3_V2; // V2.y
		//grad.host_arr[x2 + mesh_indices.startVy] += -A.z * coeff_E3_squared_norm_V2_X * coeff_E3_V2;

		//grad.host_arr[x2 + mesh_indices.startVz] += A.y* coeff_E3_squared_norm_V2_X* coeff_E3_V2; // V2.z
		//grad.host_arr[x2 + mesh_indices.startVz] += -A.x * coeff_E3_squared_norm_V2_Y * coeff_E3_V2;

		//grad.host_arr[fi + mesh_indices.startCx] += -A.z* coeff_E3_squared_norm_V0_Y * coeff_E3_V0; // C.x
		//grad.host_arr[fi + mesh_indices.startCx] += A.y * coeff_E3_squared_norm_V0_Z* coeff_E3_V0;
		//grad.host_arr[fi + mesh_indices.startCx] += -A.z * coeff_E3_squared_norm_V1_Y * coeff_E3_V1; // C.x
		//grad.host_arr[fi + mesh_indices.startCx] += A.y* coeff_E3_squared_norm_V1_Z* coeff_E3_V1;
		//grad.host_arr[fi + mesh_indices.startCx] += -A.z * coeff_E3_squared_norm_V2_Y * coeff_E3_V2; // C.x
		//grad.host_arr[fi + mesh_indices.startCx] += A.y* coeff_E3_squared_norm_V2_Z* coeff_E3_V2;

		//grad.host_arr[fi + mesh_indices.startCy] += -A.x* coeff_E3_squared_norm_V0_Z * coeff_E3_V0; // C.y
		//grad.host_arr[fi + mesh_indices.startCy] += A.z * coeff_E3_squared_norm_V0_X* coeff_E3_V0;
		//grad.host_arr[fi + mesh_indices.startCy] += -A.x * coeff_E3_squared_norm_V1_Z * coeff_E3_V1; // C.y
		//grad.host_arr[fi + mesh_indices.startCy] += A.z* coeff_E3_squared_norm_V1_X* coeff_E3_V1;
		//grad.host_arr[fi + mesh_indices.startCy] += -A.x * coeff_E3_squared_norm_V2_Z * coeff_E3_V2; // C.y
		//grad.host_arr[fi + mesh_indices.startCy] += A.z* coeff_E3_squared_norm_V2_X* coeff_E3_V2;

		//grad.host_arr[fi + mesh_indices.startCz] += -A.y* coeff_E3_squared_norm_V0_X * coeff_E3_V0; // C.z
		//grad.host_arr[fi + mesh_indices.startCz] += A.x * coeff_E3_squared_norm_V0_Y* coeff_E3_V0;
		//grad.host_arr[fi + mesh_indices.startCz] += -A.y * coeff_E3_squared_norm_V1_X * coeff_E3_V1; // C.z
		//grad.host_arr[fi + mesh_indices.startCz] += A.x* coeff_E3_squared_norm_V1_Y* coeff_E3_V1;
		//grad.host_arr[fi + mesh_indices.startCz] += -A.y * coeff_E3_squared_norm_V2_X * coeff_E3_V2; // C.z
		//grad.host_arr[fi + mesh_indices.startCz] += A.x* coeff_E3_squared_norm_V2_Y* coeff_E3_V2;
	}

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}
