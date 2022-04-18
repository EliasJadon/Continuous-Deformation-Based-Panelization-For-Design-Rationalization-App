#include "STVK.h"

STVK::STVK(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) : ObjectiveFunction{ V,F }
{
	name = "STVK";
	w = 0;
	
	shearModulus = 0.3;
	bulkModulus = 1.5;
	Cuda::AllocateMemory(dXInv, F.rows());
	//compute the area for each triangle
	igl::doublearea(restShapeV, restShapeF, restShapeArea);
	restShapeArea /= 2;

	setRestShapeFromCurrentConfiguration();
	std::cout << "\t" << name << " constructor" << std::endl;
}

STVK::~STVK() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void STVK::setRestShapeFromCurrentConfiguration() {
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		//Vertices in 3D
		Eigen::VectorXd V0_3D = restShapeV.row(restShapeF(fi, 0));
		Eigen::VectorXd V1_3D = restShapeV.row(restShapeF(fi, 1));
		Eigen::VectorXd V2_3D = restShapeV.row(restShapeF(fi, 2));
		Eigen::VectorXd e10 = V1_3D - V0_3D;
		Eigen::VectorXd e20 = V2_3D - V0_3D;

		//Flatten Vertices to 2D
		double h = e10.norm();
		double temp = e20.transpose() * e10;
		double i = temp / h;
		double j = sqrt(e20.squaredNorm() - pow(i, 2));
		Eigen::Vector2d V0_2D(0, 0);
		Eigen::Vector2d V1_2D(h, 0);
		Eigen::Vector2d V2_2D(i, j);

		//matrix that holds three edge vectors
		Eigen::Matrix2d dX;
		dX <<
			V1_2D[0], V2_2D[0],
			V1_2D[1], V2_2D[1];
		Eigen::Matrix2d inv = dX.inverse();//TODO .inverse() is baaad
		dXInv.host_arr[fi] = double_4(inv(0, 0), inv(0, 1), inv(1, 0), inv(1, 1));
	}
}

double STVK::value(Cuda::Array<double>& curr_x, const bool update) {
	double value = 0;

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const double_4 dxinv = dXInv.host_arr[fi];
		const double Area = restShapeArea[fi];
		const unsigned int v0i = restShapeF(fi,0);
		const unsigned int v1i = restShapeF(fi,1);
		const unsigned int v2i = restShapeF(fi,2);
		double_3 V0 = getV(curr_x, v0i);
		double_3 V1 = getV(curr_x, v1i);
		double_3 V2 = getV(curr_x, v2i);

		double_3 e10 = sub(V1, V0);
		double_3 e20 = sub(V2, V0);
		double dx[3][2];
		dx[0][0] = e10.x; dx[0][1] = e20.x;
		dx[1][0] = e10.y; dx[1][1] = e20.y;
		dx[2][0] = e10.z; dx[2][1] = e20.z;

		double F[3][2];
		double dxInv[2][2];
		dxInv[0][0] = dxinv.x;
		dxInv[0][1] = dxinv.y;
		dxInv[1][0] = dxinv.z;
		dxInv[1][1] = dxinv.w;
		multiply<3, 2, 2>(dx, dxInv, F);

		//compute the Green Strain = 1/2 * (F'F-I)
		double strain[2][2];
		multiplyTranspose<2, 3, 2>(F, F, strain);
		strain[0][0] -= 1; strain[1][1] -= 1;
		strain[0][0] *= 0.5;
		strain[0][1] *= 0.5;
		strain[1][0] *= 0.5;
		strain[1][1] *= 0.5;

		double energy =
			shearModulus * (pow(strain[0][0], 2) + pow(strain[1][0], 2) + pow(strain[0][1], 2) + pow(strain[1][1], 2)) +
			(bulkModulus / 2) * pow((strain[0][0] + strain[1][1]), 2);

		value += restShapeArea[fi] * energy;
		if (update) 
			Efi[fi] = energy;
	}

	if (update)
		energy_value = value;
	return value;
}

void STVK::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;
	
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const double_4 dxinv = dXInv.host_arr[fi];
		const double Area = restShapeArea[fi];
		const unsigned int v0i = restShapeF(fi,0);
		const unsigned int v1i = restShapeF(fi,1);
		const unsigned int v2i = restShapeF(fi,2);
		double_3 V0 = getV(X,v0i); 
		double_3 V1 = getV(X,v1i); 
		double_3 V2 = getV(X,v2i); 
		double_3 e10 = sub(V1, V0);
		double_3 e20 = sub(V2, V0);
		double dx[3][2];
		dx[0][0] = e10.x; dx[0][1] = e20.x;
		dx[1][0] = e10.y; dx[1][1] = e20.y;
		dx[2][0] = e10.z; dx[2][1] = e20.z;

		double F[3][2];
		double dxInv[2][2];
		dxInv[0][0] = dxinv.x;
		dxInv[0][1] = dxinv.y;
		dxInv[1][0] = dxinv.z;
		dxInv[1][1] = dxinv.w;
		multiply<3, 2, 2>(dx, dxInv, F);

		//compute the Green Strain = 1/2 * (F'F-I)
		double strain[2][2];
		multiplyTranspose<2, 3, 2>(F, F, strain);
		strain[0][0] -= 1; strain[1][1] -= 1;
		strain[0][0] *= 0.5;
		strain[0][1] *= 0.5;
		strain[1][0] *= 0.5;
		strain[1][1] *= 0.5;

		double dF_dX[6][9] = { 0 };
		dF_dX[0][0] = -dxinv.x - dxinv.z;
		dF_dX[0][1] = dxinv.x;
		dF_dX[0][2] = dxinv.z;

		dF_dX[1][0] = -dxinv.y - dxinv.w;
		dF_dX[1][1] = dxinv.y;
		dF_dX[1][2] = dxinv.w;

		dF_dX[2][3] = -dxinv.x - dxinv.z;
		dF_dX[2][4] = dxinv.x;
		dF_dX[2][5] = dxinv.z;

		dF_dX[3][3] = -dxinv.y - dxinv.w;
		dF_dX[3][4] = dxinv.y;
		dF_dX[3][5] = dxinv.w;

		dF_dX[4][6] = -dxinv.x - dxinv.z;
		dF_dX[4][7] = dxinv.x;
		dF_dX[4][8] = dxinv.z;

		dF_dX[5][6] = -dxinv.y - dxinv.w;
		dF_dX[5][7] = dxinv.y;
		dF_dX[5][8] = dxinv.w;

		double dstrain_dF[4][6] = { 0 };
		dstrain_dF[0][0] = F[0][0];
		dstrain_dF[0][2] = F[1][0];
		dstrain_dF[0][4] = F[2][0];

		dstrain_dF[1][0] = 0.5 * F[0][1];
		dstrain_dF[1][1] = 0.5 * F[0][0];
		dstrain_dF[1][2] = 0.5 * F[1][1];
		dstrain_dF[1][3] = 0.5 * F[1][0];
		dstrain_dF[1][4] = 0.5 * F[2][1];
		dstrain_dF[1][5] = 0.5 * F[2][0];

		dstrain_dF[2][0] = 0.5 * F[0][1];
		dstrain_dF[2][1] = 0.5 * F[0][0];
		dstrain_dF[2][2] = 0.5 * F[1][1];
		dstrain_dF[2][3] = 0.5 * F[1][0];
		dstrain_dF[2][4] = 0.5 * F[2][1];
		dstrain_dF[2][5] = 0.5 * F[2][0];

		dstrain_dF[3][1] = F[0][1];
		dstrain_dF[3][3] = F[1][1];
		dstrain_dF[3][5] = F[2][1];

		double dE_dJ[1][4];
		dE_dJ[0][0] = Area * (2 * shearModulus * strain[0][0] + bulkModulus * (strain[0][0] + strain[1][1]));
		dE_dJ[0][1] = Area * (2 * shearModulus * strain[0][1]);
		dE_dJ[0][2] = Area * (2 * shearModulus * strain[1][0]);
		dE_dJ[0][3] = Area * (2 * shearModulus * strain[1][1] + bulkModulus * (strain[0][0] + strain[1][1]));

		double dE_dX[1][9];
		double temp[1][6];
		multiply<1, 4, 6>(dE_dJ, dstrain_dF, temp);
		multiply<1, 6, 9>(temp, dF_dX, dE_dX);

		grad.host_arr[v0i + mesh_indices.startVx] += dE_dX[0][0];
		grad.host_arr[v1i + mesh_indices.startVx] += dE_dX[0][1];
		grad.host_arr[v2i + mesh_indices.startVx] += dE_dX[0][2];
		grad.host_arr[v0i + mesh_indices.startVy] += dE_dX[0][3];
		grad.host_arr[v1i + mesh_indices.startVy] += dE_dX[0][4];
		grad.host_arr[v2i + mesh_indices.startVy] += dE_dX[0][5];
		grad.host_arr[v0i + mesh_indices.startVz] += dE_dX[0][6];
		grad.host_arr[v1i + mesh_indices.startVz] += dE_dX[0][7];
		grad.host_arr[v2i + mesh_indices.startVz] += dE_dX[0][8];
	}

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}
