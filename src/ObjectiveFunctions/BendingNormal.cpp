#include "BendingNormal.h"
#include <igl/per_face_normals.h>

BendingNormal::BendingNormal(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixX3i& F,
	const Cuda::PenaltyFunction type) : AuxVariables{ V,F,type }	
{
	CurrV.resize(V.rows(), 3);
	name = "Bending Normal";
	w = 0;
	std::cout << "\t" << name << " constructor" << std::endl;
}

BendingNormal::~BendingNormal() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

double BendingNormal::value(Cuda::Array<double>& curr_x, const bool update)
{
	for (int vi = 0; vi < restShapeV.rows(); vi++) {
		double_3 V = getV(curr_x, vi);
		CurrV.row(vi) = Eigen::RowVector3d(V.x, V.y, V.z);
	}
	igl::per_face_normals(CurrV, (Eigen::MatrixX3i)restShapeF, normals);
	double value = 0;
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		double d_normals = (normals.row(f1) - normals.row(f0)).squaredNorm();
		value += restAreaPerHinge[hi] * weight_PerHinge.host_arr[hi] *
			Phi(d_normals, Sigmoid_PerHinge.host_arr[hi], penaltyFunction);
	}

	if (update)
		energy_value = value;
	return value;
}

void BendingNormal::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++) {
		grad.host_arr[i] = 0;
	}

	for (int vi = 0; vi < restShapeV.rows(); vi++) {
		double_3 V = getV(X, vi);
		CurrV.row(vi) = Eigen::RowVector3d(V.x, V.y, V.z);
	}
	igl::per_face_normals(CurrV, (Eigen::MatrixX3i)restShapeF, normals);

	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		double d_normals = (normals.row(f1) - normals.row(f0)).squaredNorm();
		
		Eigen::Matrix<double, 6, 12> n_x = dN_dx_perhinge(hi);
		Eigen::Matrix<double, 1, 12> dE_dx =
			restAreaPerHinge[hi]
			* weight_PerHinge.host_arr[hi]
			* dPhi_dm(d_normals, Sigmoid_PerHinge.host_arr[hi], penaltyFunction)
			* dm_dN(hi).transpose()
			* n_x;

		for (int xi = 0; xi < 4; xi++)
			for (int xyz = 0; xyz < 3; ++xyz)
				grad.host_arr[x_GlobInd(xi, hi) + (xyz * restShapeV.rows())] += dE_dx(xi * 3 + xyz);
	}

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}


Eigen::Matrix< double, 6, 1> BendingNormal::dm_dN(int hi) {
	// m = ||n1 - n0||^2
	// m = (n1.x - n0.x)^2 + (n1.y - n0.y)^2 + (n1.z - n0.z)^2
	int f0 = hinges_faceIndex[hi](0);
	int f1 = hinges_faceIndex[hi](1);
	Eigen::Matrix< double, 6, 1> grad;
	grad <<
		-2 * (normals(f1, 0) - normals(f0, 0)),	//n0.x
		-2 * (normals(f1, 1) - normals(f0, 1)), //n0.y
		-2 * (normals(f1, 2) - normals(f0, 2)), //n0.z
		2 * (normals(f1, 0) - normals(f0, 0)),	//n1.x
		2 * (normals(f1, 1) - normals(f0, 1)),	//n1.y
		2 * (normals(f1, 2) - normals(f0, 2));	//n1.z
	return grad;
}

Eigen::Matrix<double, 6, 12> BendingNormal::dN_dx_perhinge(int hi) {
	int f0 = hinges_faceIndex[hi](0);
	int f1 = hinges_faceIndex[hi](1);
	Eigen::Matrix<double, 3, 9> n0_x = dN_dx_perface(f0);
	Eigen::Matrix<double, 3, 9> n1_x = dN_dx_perface(f1);
	Eigen::Matrix<double, 6, 12> n_x;
	n_x.setZero();

	n_x.block<3, 3>(0, 0) = n0_x.block<3, 3>(0, x0_LocInd(hi, 0) * 3);
	n_x.block<3, 3>(0, 3) = n0_x.block<3, 3>(0, x1_LocInd(hi, 0) * 3);
	n_x.block<3, 3>(0, 6) = n0_x.block<3, 3>(0, x2_LocInd(hi, 0) * 3);

	n_x.block<3, 3>(3, 0) = n1_x.block<3, 3>(0, x0_LocInd(hi, 1) * 3);
	n_x.block<3, 3>(3, 3) = n1_x.block<3, 3>(0, x1_LocInd(hi, 1) * 3);
	n_x.block<3, 3>(3, 9) = n1_x.block<3, 3>(0, x3_LocInd(hi, 1) * 3);

	return n_x;
}

Eigen::Matrix<double, 3, 9> BendingNormal::dN_dx_perface(int fi) {
	// e1 = v1-v0
	// e2 = v2-v0
	//
	// N = e1 x e2
	// N.x = (y1-y0)*(z2-z0)-(z1-z0)*(y2-y0)
	// N.y = (z1-z0)*(x2-x0)-(x1-x0)*(z2-z0)
	// N.z = (x1-x0)*(y2-y0)-(y1-y0)*(x2-x0)
	//
	// NormalizedN = N / norm

	Eigen::Vector3d e0 = CurrV.row(restShapeF(fi, 1)) - CurrV.row(restShapeF(fi, 0));
	Eigen::Vector3d e1 = CurrV.row(restShapeF(fi, 2)) - CurrV.row(restShapeF(fi, 0));
	Eigen::Vector3d N = e0.cross(e1);
	double norm = N.norm();
	Eigen::Matrix<double, 9, 3> jacobian_N;
	Eigen::Matrix<double, 9, 1> grad_norm;
	jacobian_N <<
		0, -e0(2) + e1(2), -e1(1) + e0(1),	//x0
		-e1(2) + e0(2), 0, -e0(0) + e1(0),	//y0
		-e0(1) + e1(1), -e1(0) + e0(0), 0,	//z0
		0, -e1(2), e1(1),	//x1
		e1(2), 0, -e1(0),	//y1
		-e1(1), e1(0), 0,	//z1
		0, e0(2), -e0(1),	//x2
		-e0(2), 0, e0(0),	//y2
		e0(1), -e0(0), 0;			//z2

	grad_norm = (N(0) * jacobian_N.col(0) + N(1) * jacobian_N.col(1) + N(2) * jacobian_N.col(2)) / norm;

	Eigen::Matrix<double, 3, 9> jacobian_normalizedN;
	for (int i = 0; i < 3; i++)
		jacobian_normalizedN.row(i) = (jacobian_N.col(i) / norm) - ((grad_norm * N(i)) / pow(norm, 2));
	return jacobian_normalizedN;
}

int BendingNormal::x_GlobInd(int index, int hi) {
	if (index == 0)
		return x0_GlobInd(hi);
	if (index == 1)
		return x1_GlobInd(hi);
	if (index == 2)
		return x2_GlobInd(hi);
	if (index == 3)
		return x3_GlobInd(hi);
}
