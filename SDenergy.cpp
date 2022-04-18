#include "SDenergy.h"


SDenergy::SDenergy(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) : ObjectiveFunction{ V,F } 
{
	name = "Symmetric Dirichlet";
	w = 0.5;
	Eigen::MatrixX3d D1cols, D2cols;
	OptimizationUtils::computeSurfaceGradientPerFace(restShapeV, restShapeF, D1cols, D2cols);
	igl::doublearea(restShapeV, restShapeF, restShapeArea);
	restShapeArea /= 2;

	Cuda::AllocateMemory(D1d, D1cols.rows());
	Cuda::AllocateMemory(D2d, D2cols.rows());
	for (int i = 0; i < D1d.size; i++)
		D1d.host_arr[i] = double_3(D1cols(i, 0), D1cols(i, 1), D1cols(i, 2));
	for (int i = 0; i < D2d.size; i++)
		D2d.host_arr[i] = double_3(D2cols(i, 0), D2cols(i, 1), D2cols(i, 2));

	std::cout << "\t" << name << " constructor" << std::endl;
}

SDenergy::~SDenergy() {
	Cuda::FreeMemory(D1d);
	Cuda::FreeMemory(D2d);
	std::cout << "\t" << name << " destructor" << std::endl;
}

double SDenergy::value(Cuda::Array<double>& curr_x, const bool update) {
	double value = 0;

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		int v0_index = restShapeF(fi, 0);
		int v1_index = restShapeF(fi, 1);
		int v2_index = restShapeF(fi, 2);
		double_3 V0 = getV(curr_x, v0_index);
		double_3 V1 = getV(curr_x, v1_index);
		double_3 V2 = getV(curr_x, v2_index);
		double_3 e10 = sub(V1, V0);
		double_3 e20 = sub(V2, V0);
		double_3 B1 = normalize(e10);
		double_3 B2 = normalize(cross(cross(B1, e20), B1));
		double_3 Xi = double_3(dot(V0, B1), dot(V1, B1), dot(V2, B1));
		double_3 Yi = double_3(dot(V0, B2), dot(V1, B2), dot(V2, B2));
		//prepare jacobian		
		const double a = dot(D1d.host_arr[fi], Xi);
		const double b = dot(D1d.host_arr[fi], Yi);
		const double c = dot(D2d.host_arr[fi], Xi);
		const double d = dot(D2d.host_arr[fi], Yi);
		const double detJ = a * d - b * c;
		const double detJ2 = detJ * detJ;
		const double a2 = a * a;
		const double b2 = b * b;
		const double c2 = c * c;
		const double d2 = d * d;
		double energy = 0.5 * (1 + 1 / detJ2) * (a2 + b2 + c2 + d2);
		value += restShapeArea[fi] * energy;
		if (update)
			Efi[fi] = energy;
	}
		
	if (update)
		energy_value = value;
	return value;
}

void SDenergy::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const int v0_index = restShapeF(fi, 0);
		const int v1_index = restShapeF(fi, 1);
		const int v2_index = restShapeF(fi, 2);
		double_3 V0 = getV(X, v0_index);
		double_3 V1 = getV(X, v1_index);
		double_3 V2 = getV(X, v2_index);
		double_3 e10 = sub(V1, V0);
		double_3 e20 = sub(V2, V0);
		double_3 B1 = normalize(e10);
		double_3 B2 = normalize(cross(cross(B1, e20), B1));
		double_3 Xi = double_3(dot(V0, B1), dot(V1, B1), dot(V2, B1));
		double_3 Yi = double_3(dot(V0, B2), dot(V1, B2), dot(V2, B2));
		//prepare jacobian		
		const double a = dot(D1d.host_arr[fi], Xi);
		const double b = dot(D1d.host_arr[fi], Yi);
		const double c = dot(D2d.host_arr[fi], Xi);
		const double d = dot(D2d.host_arr[fi], Yi);
		const double detJ = a * d - b * c;
		const double det2 = pow(detJ, 2);
		const double a2 = pow(a, 2);
		const double b2 = pow(b, 2);
		const double c2 = pow(c, 2);
		const double d2 = pow(d, 2);
		const double det3 = pow(detJ, 3);
		const double Fnorm = a2 + b2 + c2 + d2;

		double_4 de_dJ = double_4(
			restShapeArea[fi] * (a + a / det2 - d * Fnorm / det3),
			restShapeArea[fi] * (b + b / det2 + c * Fnorm / det3),
			restShapeArea[fi] * (c + c / det2 + b * Fnorm / det3),
			restShapeArea[fi] * (d + d / det2 - a * Fnorm / det3)
		);
		double Norm_e10_3 = pow(norm(e10), 3);
		double_3 B2_b2 = cross(cross(e10, e20), e10);
		double Norm_B2 = norm(B2_b2);
		double Norm_B2_2 = pow(Norm_B2, 2);
		double_3 B2_dxyz0, B2_dxyz1;
		double B2_dnorm0, B2_dnorm1;
		double_3 db1_dX, db2_dX, XX, YY;
		double_4 dj_dx;

		B2_dxyz0 = double_3(-e10.y * e20.y - e10.z * e20.z, 2 * e10.x * e20.y - e10.y * e20.x, -e10.z * e20.x + 2 * e10.x * e20.z);
		B2_dxyz1 = double_3(pow(e10.y, 2) + pow(e10.z, 2), -e10.x * e10.y, -e10.x * e10.z);
		B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
		B2_dnorm1 = dot(B2_b2, B2_dxyz1) / Norm_B2;
		db2_dX = double_3(
			-((B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.x * Norm_B2 - B2_b2.x * B2_dnorm1) / Norm_B2_2),
			-((B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.y * Norm_B2 - B2_b2.y * B2_dnorm1) / Norm_B2_2),
			-((B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.z * Norm_B2 - B2_b2.z * B2_dnorm1) / Norm_B2_2)
		);
		db1_dX = double_3((-(pow(e10.y, 2) + pow(e10.z, 2)) / Norm_e10_3), ((e10.y * e10.x) / Norm_e10_3), ((e10.z * e10.x) / Norm_e10_3));
		XX = double_3(dot(V0, db1_dX) + B1.x, dot(V1, db1_dX), dot(V2, db1_dX));
		YY = double_3(dot(V0, db2_dX) + B2.x, dot(V1, db2_dX), dot(V2, db2_dX));
		dj_dx = double_4(
			dot(D1d.host_arr[fi], XX),
			dot(D1d.host_arr[fi], YY),
			dot(D2d.host_arr[fi], XX),
			dot(D2d.host_arr[fi], YY)
		);
		grad.host_arr[v0_index + mesh_indices.startVx] += dot4(de_dJ, dj_dx);


		B2_dxyz0 = double_3(-e10.y * e20.y - e10.z * e20.z, 2 * e10.x * e20.y - e10.y * e20.x, -e10.z * e20.x + 2 * e10.x * e20.z);
		B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
		db2_dX = double_3(
			(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
		);
		db1_dX = double_3(-(-(pow(e10.y, 2) + pow(e10.z, 2)) / Norm_e10_3), -((e10.y * e10.x) / Norm_e10_3), -((e10.z * e10.x) / Norm_e10_3));
		XX = double_3(dot(V0, db1_dX), dot(V1, db1_dX) + B1.x, dot(V2, db1_dX));
		YY = double_3(dot(V0, db2_dX), dot(V1, db2_dX) + B2.x, dot(V2, db2_dX));
		dj_dx = double_4(
			dot(D1d.host_arr[fi], XX),
			dot(D1d.host_arr[fi], YY),
			dot(D2d.host_arr[fi], XX),
			dot(D2d.host_arr[fi], YY)
		);
		grad.host_arr[v1_index + mesh_indices.startVx] += dot4(de_dJ, dj_dx);




		B2_dxyz0 = double_3(pow(e10.y, 2) + pow(e10.z, 2), -e10.x * e10.y, -e10.x * e10.z);
		B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
		db2_dX = double_3(
			(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
		);
		XX = double_3(0, 0, B1.x);
		YY = double_3(dot(V0, db2_dX), dot(V1, db2_dX), dot(V2, db2_dX) + B2.x);
		dj_dx = double_4(
			dot(D1d.host_arr[fi], XX),
			dot(D1d.host_arr[fi], YY),
			dot(D2d.host_arr[fi], XX),
			dot(D2d.host_arr[fi], YY)
		);
		grad.host_arr[v2_index + mesh_indices.startVx] += dot4(de_dJ, dj_dx);


		B2_dxyz0 = double_3(-e10.x * e20.y + 2 * e10.y * e20.x, -e10.z * e20.z - e20.x * e10.x, 2 * e10.y * e20.z - e10.z * e20.y);
		B2_dxyz1 = double_3(-e10.y * e10.x, pow(e10.z, 2) + pow(e10.x, 2), -e10.z * e10.y);
		B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
		B2_dnorm1 = dot(B2_b2, B2_dxyz1) / Norm_B2;
		db2_dX = double_3(
			-((B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.x * Norm_B2 - B2_b2.x * B2_dnorm1) / Norm_B2_2),
			-((B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.y * Norm_B2 - B2_b2.y * B2_dnorm1) / Norm_B2_2),
			-((B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.z * Norm_B2 - B2_b2.z * B2_dnorm1) / Norm_B2_2)
		);
		db1_dX = double_3(((e10.y * e10.x) / Norm_e10_3), (-(pow(e10.x, 2) + pow(e10.z, 2)) / Norm_e10_3), ((e10.z * e10.y) / Norm_e10_3));
		XX = double_3(dot(V0, db1_dX) + B1.y, dot(V1, db1_dX), dot(V2, db1_dX));
		YY = double_3(dot(V0, db2_dX) + B2.y, dot(V1, db2_dX), dot(V2, db2_dX));
		dj_dx = double_4(
			dot(D1d.host_arr[fi], XX),
			dot(D1d.host_arr[fi], YY),
			dot(D2d.host_arr[fi], XX),
			dot(D2d.host_arr[fi], YY)
		);
		grad.host_arr[v0_index + mesh_indices.startVy] += dot4(de_dJ, dj_dx);



		B2_dxyz0 = double_3(-e10.x * e20.y + 2 * e10.y * e20.x, -e10.z * e20.z - e20.x * e10.x, 2 * e10.y * e20.z - e10.z * e20.y);
		B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
		db2_dX = double_3(
			(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
		);
		db1_dX = double_3(-((e10.y * e10.x) / Norm_e10_3), -(-(pow(e10.x, 2) + pow(e10.z, 2)) / Norm_e10_3), -((e10.z * e10.y) / Norm_e10_3));
		XX = double_3(dot(V0, db1_dX), dot(V1, db1_dX) + B1.y, dot(V2, db1_dX));
		YY = double_3(dot(V0, db2_dX), dot(V1, db2_dX) + B2.y, dot(V2, db2_dX));
		dj_dx = double_4(
			dot(D1d.host_arr[fi], XX),
			dot(D1d.host_arr[fi], YY),
			dot(D2d.host_arr[fi], XX),
			dot(D2d.host_arr[fi], YY)
		);
		grad.host_arr[v1_index + mesh_indices.startVy] += dot4(de_dJ, dj_dx);



		B2_dxyz0 = double_3(-e10.y * e10.x, pow(e10.z, 2) + pow(e10.x, 2), -e10.z * e10.y);
		B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
		db2_dX = double_3(
			(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
		);
		XX = double_3(0, 0, B1.y);
		YY = double_3(dot(V0, db2_dX), dot(V1, db2_dX), dot(V2, db2_dX) + B2.y);
		dj_dx = double_4(
			dot(D1d.host_arr[fi], XX),
			dot(D1d.host_arr[fi], YY),
			dot(D2d.host_arr[fi], XX),
			dot(D2d.host_arr[fi], YY)
		);
		grad.host_arr[v2_index + mesh_indices.startVy] += dot4(de_dJ, dj_dx);



		B2_dxyz0 = double_3(2 * e10.z * e20.x - e10.x * e20.z, -e10.y * e20.z + 2 * e10.z * e20.y, -e10.x * e20.x - e10.y * e20.y);
		B2_dxyz1 = double_3(-e10.x * e10.z, -e10.z * e10.y, pow(e10.x, 2) + pow(e10.y, 2));
		B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
		B2_dnorm1 = dot(B2_b2, B2_dxyz1) / Norm_B2;
		db2_dX = double_3(
			-((B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.x * Norm_B2 - B2_b2.x * B2_dnorm1) / Norm_B2_2),
			-((B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.y * Norm_B2 - B2_b2.y * B2_dnorm1) / Norm_B2_2),
			-((B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.z * Norm_B2 - B2_b2.z * B2_dnorm1) / Norm_B2_2)
		);
		db1_dX = double_3(((e10.z * e10.x) / Norm_e10_3), ((e10.z * e10.y) / Norm_e10_3), (-(pow(e10.x, 2) + pow(e10.y, 2)) / Norm_e10_3));
		XX = double_3(dot(V0, db1_dX) + B1.z, dot(V1, db1_dX), dot(V2, db1_dX));
		YY = double_3(dot(V0, db2_dX) + B2.z, dot(V1, db2_dX), dot(V2, db2_dX));
		dj_dx = double_4(
			dot(D1d.host_arr[fi], XX),
			dot(D1d.host_arr[fi], YY),
			dot(D2d.host_arr[fi], XX),
			dot(D2d.host_arr[fi], YY)
		);
		grad.host_arr[v0_index + mesh_indices.startVz] += dot4(de_dJ, dj_dx);




		B2_dxyz0 = double_3(2 * e10.z * e20.x - e10.x * e20.z, -e10.y * e20.z + 2 * e10.z * e20.y, -e10.x * e20.x - e10.y * e20.y);
		B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
		db2_dX = double_3(
			(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
		);
		db1_dX = double_3(-((e10.z * e10.x) / Norm_e10_3), -((e10.z * e10.y) / Norm_e10_3), -(-(pow(e10.x, 2) + pow(e10.y, 2)) / Norm_e10_3));
		XX = double_3(dot(V0, db1_dX), dot(V1, db1_dX) + B1.z, dot(V2, db1_dX));
		YY = double_3(dot(V0, db2_dX), dot(V1, db2_dX) + B2.z, dot(V2, db2_dX));
		dj_dx = double_4(
			dot(D1d.host_arr[fi], XX),
			dot(D1d.host_arr[fi], YY),
			dot(D2d.host_arr[fi], XX),
			dot(D2d.host_arr[fi], YY)
		);
		grad.host_arr[v1_index + mesh_indices.startVz] += dot4(de_dJ, dj_dx);



		B2_dxyz0 = double_3(-e10.x * e10.z, -e10.z * e10.y, pow(e10.x, 2) + pow(e10.y, 2));
		B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
		db2_dX = double_3(
			(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
			(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
		);
		XX = double_3(0, 0, B1.z);
		YY = double_3(dot(V0, db2_dX), dot(V1, db2_dX), dot(V2, db2_dX) + B2.z);
		dj_dx = double_4(
			dot(D1d.host_arr[fi], XX),
			dot(D1d.host_arr[fi], YY),
			dot(D2d.host_arr[fi], XX),
			dot(D2d.host_arr[fi], YY)
		);
		grad.host_arr[v2_index + mesh_indices.startVz] += dot4(de_dJ, dj_dx);
	}

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}
