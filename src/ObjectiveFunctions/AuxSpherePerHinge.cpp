#include "AuxSpherePerHinge.h"

AuxSpherePerHinge::AuxSpherePerHinge(
	const Eigen::MatrixXd& V, 
	const Eigen::MatrixX3i& F,
	const Cuda::PenaltyFunction type) : AuxVariables{V,F,type}
{
	name = "Aux Sphere Per Hinge";
	std::cout << "\t" << name << " constructor" << std::endl;
}

AuxSpherePerHinge::~AuxSpherePerHinge() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

double AuxSpherePerHinge::value(Cuda::Array<double>& curr_x, const bool update)
{	
	double value = 0;

	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi][0];
		int f1 = hinges_faceIndex[hi][1];
		double R0 = getR(curr_x, f0); 
		double R1 = getR(curr_x, f1); 
		double_3 C0 = getC(curr_x, f0);
		double_3 C1 = getC(curr_x, f1);
			
		double d_center = squared_norm(sub(C1, C0));
		double d_radius = pow(R1 - R0, 2);
		value += w1 * restAreaPerHinge[hi] * weight_PerHinge.host_arr[hi] *
			Phi(d_center + d_radius, Sigmoid_PerHinge.host_arr[hi], penaltyFunction);
	}
	
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const unsigned int x0 = restShapeF(fi,0);
		const unsigned int x1 = restShapeF(fi,1);
		const unsigned int x2 = restShapeF(fi,2);
		double_3 V0 = getV(curr_x, x0);
		double_3 V1 = getV(curr_x, x1);
		double_3 V2 = getV(curr_x, x2);
		double_3 C = getC(curr_x, fi);
		double R = getR(curr_x, fi);
		
		double res =
			pow(squared_norm(sub(V0, C)) - pow(R, 2), 2) +
			pow(squared_norm(sub(V1, C)) - pow(R, 2), 2) +
			pow(squared_norm(sub(V2, C)) - pow(R, 2), 2);

		value += w2 * res;
	}
	if (update)
		energy_value = value;
	return value;
}

void AuxSpherePerHinge::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;

	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi][0];
		int f1 = hinges_faceIndex[hi][1];
		
		double R0 = getR(X, f0);
		double R1 = getR(X, f1);
		double_3 C0 = getC(X, f0); 
		double_3 C1 = getC(X, f1); 
		
		double d_center = squared_norm(sub(C1, C0));
		double d_radius = pow(R1 - R0, 2);
		double coeff = 2 * w1 * restAreaPerHinge[hi] * weight_PerHinge.host_arr[hi] *
			dPhi_dm(d_center + d_radius, Sigmoid_PerHinge.host_arr[hi], penaltyFunction);

		grad.host_arr[f0 + mesh_indices.startCx] += (C0.x - C1.x) * coeff; //C0.x
		grad.host_arr[f0 + mesh_indices.startCy] += (C0.y - C1.y) * coeff;	//C0.y
		grad.host_arr[f0 + mesh_indices.startCz] += (C0.z - C1.z) * coeff;	//C0.z
		grad.host_arr[f1 + mesh_indices.startCx] += (C1.x - C0.x) * coeff;	//C1.x
		grad.host_arr[f1 + mesh_indices.startCy] += (C1.y - C0.y) * coeff;	//C1.y
		grad.host_arr[f1 + mesh_indices.startCz] += (C1.z - C0.z) * coeff;	//C1.z
		grad.host_arr[f0 + mesh_indices.startR] += (R0 - R1) * coeff;		//r0
		grad.host_arr[f1 + mesh_indices.startR] += (R1 - R0) * coeff;		//r1
	}
	

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const unsigned int x0 = restShapeF(fi, 0);
		const unsigned int x1 = restShapeF(fi, 1);
		const unsigned int x2 = restShapeF(fi, 2);
		
		double_3 V0 = getV(X, x0); 
		double_3 V1 = getV(X, x1); 
		double_3 V2 = getV(X, x2); 
		double_3 C = getC(X, fi); 
		double R = getR(X, fi); 
		
		double coeff = w2 * 4;
		double E0 = coeff * (squared_norm(sub(V0, C)) - pow(R, 2));
		double E1 = coeff * (squared_norm(sub(V1, C)) - pow(R, 2));
		double E2 = coeff * (squared_norm(sub(V2, C)) - pow(R, 2));

		grad.host_arr[x0 + mesh_indices.startVx] += E0 * (V0.x - C.x); // V0x
		grad.host_arr[x0 + mesh_indices.startVy] += E0 * (V0.y - C.y); // V0y
		grad.host_arr[x0 + mesh_indices.startVz] += E0 * (V0.z - C.z); // V0z
		grad.host_arr[x1 + mesh_indices.startVx] += E1 * (V1.x - C.x); // V1x
		grad.host_arr[x1 + mesh_indices.startVy] += E1 * (V1.y - C.y); // V1y
		grad.host_arr[x1 + mesh_indices.startVz] += E1 * (V1.z - C.z); // V1z
		grad.host_arr[x2 + mesh_indices.startVx] += E2 * (V2.x - C.x); // V2x
		grad.host_arr[x2 + mesh_indices.startVy] += E2 * (V2.y - C.y); // V2y
		grad.host_arr[x2 + mesh_indices.startVz] += E2 * (V2.z - C.z); // V2z
		grad.host_arr[fi + mesh_indices.startCx] +=
			(E0 * (C.x - V0.x)) +
				(E1 * (C.x - V1.x)) +
				(E2 * (C.x - V2.x)); // Cx
		grad.host_arr[fi + mesh_indices.startCy] +=
			(E0 * (C.y - V0.y)) +
				(E1 * (C.y - V1.y)) +
				(E2 * (C.y - V2.y)); // Cy
		grad.host_arr[fi + mesh_indices.startCz] +=
			(E0 * (C.z - V0.z)) +
				(E1 * (C.z - V1.z)) +
				(E2 * (C.z - V2.z)); // Cz
		grad.host_arr[fi + mesh_indices.startR] +=
			(E0 * (-1) * R) +
				(E1 * (-1) * R) +
				(E2 * (-1) * R); //r
	}

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}
