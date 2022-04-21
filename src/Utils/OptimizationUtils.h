#pragma once

#include <direct.h>
#include <iostream>
#include <igl/doublearea.h>
#include <igl/local_basis.h>
#include <igl/boundary_loop.h>
#include <igl/per_face_normals.h>
#include <windows.h>
#include <Eigen/sparse>
#include <igl/vertex_triangle_adjacency.h>
#include <chrono>
#include <igl/triangle_triangle_adjacency.h>
#include <set>
#include <igl/PI.h>


class double_3 {
public:
	double x, y, z;
	double_3(double x, double y, double z) :x{ x }, y{ y }, z{ z }{};
	double_3() :x{ 0 }, y{ 0 }, z{ 0 }{};
};

class double_4 {
public:
	double x, y, z, w;
	double_4(double x, double y, double z, double w) :x{ x }, y{ y }, z{ z }, w{ w }{};
	double_4() :x{ 0 }, y{ 0 }, z{ 0 }, w{ 0 }{};
};


static double dot4(const double_4 a, const double_4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template<int N> void multiply(
	double_3 mat1,
	double mat2[3][N],
	double res[N])
{
	for (int i = 0; i < N; i++) {
		res[i] = mat1.x * mat2[0][i] + mat1.y * mat2[1][i] + mat1.z * mat2[2][i];
	}
}

template<int R1, int C1_R2, int C2> void multiply(
	double mat1[R1][C1_R2],
	double mat2[C1_R2][C2],
	double res[R1][C2])
{
	int i, j, k;
	for (i = 0; i < R1; i++) {
		for (j = 0; j < C2; j++) {
			res[i][j] = 0;
			for (k = 0; k < C1_R2; k++)
				res[i][j] += mat1[i][k] * mat2[k][j];
		}
	}
}
template<int R1, int C1_R2, int C2> void multiplyTranspose(
	double mat1[C1_R2][R1],
	double mat2[C1_R2][C2],
	double res[R1][C2])
{
	int i, j, k;
	for (i = 0; i < R1; i++) {
		for (j = 0; j < C2; j++) {
			res[i][j] = 0;
			for (k = 0; k < C1_R2; k++)
				res[i][j] += mat1[k][i] * mat2[k][j];
		}
	}
}

template<int N> void multiply(
	double_4 mat1,
	double mat2[4][N],
	double res[N])
{
	for (int i = 0; i < N; i++) {
		res[i] =
			mat1.x * mat2[0][i] +
			mat1.y * mat2[1][i] +
			mat1.z * mat2[2][i] +
			mat1.w * mat2[3][i];
	}
}


static double_3 sub(const double_3 a, const double_3 b)
{
	return double_3(a.x - b.x, a.y - b.y, a.z - b.z);
}
static double_3 sub(const double_3 a, const Eigen::RowVector3d b)
{
	return double_3(a.x - b(0), a.y - b(1), a.z - b(2));
}
static double_3 add(double_3 a, double_3 b)
{
	return double_3(a.x + b.x, a.y + b.y, a.z + b.z);
}
static double dot(const double_3 a, const double_3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
static double_3 mul(const double a, const double_3 b)
{
	return double_3(a * b.x, a * b.y, a * b.z);
}
static double squared_norm(const double_3 a)
{
	return dot(a, a);
}
static double norm(const double_3 a)
{
	return sqrt(squared_norm(a));
}
static double_3 normalize(const double_3 a)
{
	return mul(1.0f / norm(a), a);
}
static double_3 cross(const double_3 a, const double_3 b)
{
	return double_3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}


namespace Cuda
{
	enum PenaltyFunction { QUADRATIC, EXPONENTIAL, SIGMOID };
	enum OptimizerType { Gradient_Descent, Adam };

	template <typename T> struct Array
	{
		unsigned int size;
		T* host_arr;
	};

	template<typename T> void FreeMemory(Cuda::Array<T>& a)
	{
		delete[] a.host_arr;
	}

	template<typename T> void AllocateMemory(Cuda::Array<T>& a, const unsigned int size)
	{
		if (size < 0) {
			std::cout << "The size isn't positive!\n";
			exit(1);
		}
		a.size = size;
		a.host_arr = new T[size];
		if (a.host_arr == NULL) {
			std::cout << "Allocation Failed!!!\n";
			exit(1);
		}
	}

	struct indices {
		unsigned int
			startVx, startVy, startVz,
			startNx, startNy, startNz,
			startCx, startCy, startCz,
			startR,
			num_vertices, num_faces, num_hinges;
	};

	static void initIndices(indices& I,const unsigned int F,const unsigned int V,const unsigned int H) 
	{
		I.num_vertices = V;
		I.num_faces = F;
		I.num_hinges = H;
		I.startVx = 0 * V + 0 * F;
		I.startVy = 1 * V + 0 * F;
		I.startVz = 2 * V + 0 * F;
		I.startNx = 3 * V + 0 * F;
		I.startNy = 3 * V + 1 * F;
		I.startNz = 3 * V + 2 * F;
		I.startCx = 3 * V + 3 * F;
		I.startCy = 3 * V + 4 * F;
		I.startCz = 3 * V + 5 * F;
		I.startR = 3 * V + 6 * F;
	}

}

namespace OptimizationUtils
{
	enum InitSphereAuxVariables { SPHERE_FIT, MODEL_CENTER_POINT, MINUS_NORMALS };
	enum LineSearch { GRADIENT_NORM, FUNCTION_VALUE, CONSTANT_STEP };

	static std::vector<int> FaceToHinge_indices(
		const std::vector<Eigen::Vector2d>& HingeToFace_indices, 
		const std::vector<int>& faces_indices,
		const int fi)
	{
		const int numHinges = HingeToFace_indices.size();
		std::vector<int> hingesPerFace = {};
		for (int hi = 0; hi < numHinges; hi++) {
			const int f1 = HingeToFace_indices[hi](0);
			const int f2 = HingeToFace_indices[hi](1);
			if (fi == f1) {
				for (int second_f : faces_indices)
					if(f2 == second_f)
						hingesPerFace.push_back(hi);
			}
			else if (fi == f2) {
				for (int second_f : faces_indices)
					if (f1 == second_f)
						hingesPerFace.push_back(hi);
			}
				
		} 
		assert(hingesPerFace.size() <= 3 && hingesPerFace.size() >= 0);
		return hingesPerFace;
	}

	static int VertexToHinge_indices(
		const Eigen::VectorXi& x0,
		const Eigen::VectorXi& x1,
		const std::vector<int>& vertex_indices,
		const int vi)
	{
		for (int i = 0; i < x0.size(); i++) {
			if (x0[i] == vi) {
				for (int i2 : vertex_indices) {
					if (i2 == x1[i])
						return i;
				}
			}
		}
		return -1;
	}

	static int getNumberOfHinges(const Eigen::MatrixX3i restShapeF) {
		std::vector<std::vector<std::vector<int>>> TT;
		igl::triangle_triangle_adjacency(restShapeF, TT);
		assert(TT.size() == restShapeF.rows());

		int num_hinges = 0;
		///////////////////////////////////////////////////////////
		//Part 1 - Find unique hinges
		for (int fi = 0; fi < TT.size(); fi++) {
			std::vector< std::vector<int>> CurrFace = TT[fi];
			assert(CurrFace.size() == 3 && "Each face should be a triangle (not square for example)!");
			for (std::vector<int> hinge : CurrFace) {
				if (hinge.size() == 1) {
					//add this "hinge"
					int FaceIndex1 = fi;
					int FaceIndex2 = hinge[0];

					if (FaceIndex2 < FaceIndex1) {
						//Skip
						//This hinge already exists!
						//Empty on purpose
					}
					else {
						num_hinges++;
					}
				}
				else if (hinge.size() == 0) {
					//Skip
					//This triangle has no another adjacent triangle on that edge
					//Empty on purpose
				}
				else {
					//We shouldn't get here!
					//The mesh is invalid
					assert("Each triangle should have only one adjacent triangle on each edge!");
				}

			}
		}
		return num_hinges;
	}

	static void computeSurfaceGradientPerFace(const Eigen::MatrixX3d &V, const Eigen::MatrixX3i &F, Eigen::MatrixX3d &D1, Eigen::MatrixX3d &D2)
	{
		Eigen::MatrixX3d F1, F2, F3;
		igl::local_basis(V, F, F1, F2, F3);
		const int Fn = F.rows();  const int vn = V.rows();

		Eigen::MatrixX3d Dx(Fn,3), Dy(Fn, 3), Dz(Fn, 3);
		Eigen::MatrixX3d fN; igl::per_face_normals(V, F, fN);
		Eigen::VectorXd Ar; igl::doublearea(V, F, Ar);
		Eigen::PermutationMatrix<3> perm;

		Eigen::Vector3i Pi;
		Pi << 1, 2, 0;
		Eigen::PermutationMatrix<3> P = Eigen::PermutationMatrix<3>(Pi);

		for (int i = 0; i < Fn; i++) {
			// renaming indices of vertices of triangles for convenience
			int i1 = F(i, 0);
			int i2 = F(i, 1);
			int i3 = F(i, 2);

			// #F x 3 matrices of triangle edge vectors, named after opposite vertices
			Eigen::Matrix3d e;
			e.col(0) = V.row(i2) - V.row(i1);
			e.col(1) = V.row(i3) - V.row(i2);
			e.col(2) = V.row(i1) - V.row(i3);;

			Eigen::Vector3d Fni = fN.row(i);
			double Ari = Ar(i);

			//grad3_3f(:,[3*i,3*i-2,3*i-1])=[0,-Fni(3), Fni(2);Fni(3),0,-Fni(1);-Fni(2),Fni(1),0]*e/(2*Ari);
			Eigen::Matrix3d n_M;
			n_M << 0, -Fni(2), Fni(1), Fni(2), 0, -Fni(0), -Fni(1), Fni(0), 0;
			Eigen::VectorXi R(3); R << 0, 1, 2;
			Eigen::VectorXi C(3); C << 3 * i + 2, 3 * i, 3 * i + 1;
			Eigen::Matrix3d res = ((1. / Ari)*(n_M*e))*P;

			Dx.row(i) = res.row(0);
			Dy.row(i) = res.row(1);
			Dz.row(i) = res.row(2);
		}
		D1 = F1.col(0).asDiagonal()*Dx + F1.col(1).asDiagonal()*Dy + F1.col(2).asDiagonal()*Dz;
		D2 = F2.col(0).asDiagonal()*Dx + F2.col(1).asDiagonal()*Dy + F2.col(2).asDiagonal()*Dz;
	}
	
	static std::string ProjectPath() {
		char buffer[MAX_PATH];
		GetModuleFileName(NULL, buffer, MAX_PATH);
		std::string::size_type pos = std::string(buffer).find("\\MappingsLab\\");
		return std::string(buffer).substr(0, pos + 11 + 2);
	}

	static std::vector<int> temp_get_one_ring_vertices_per_vertex(const Eigen::MatrixXi& F, const std::vector<int>& OneRingFaces) {
		std::vector<int> vertices;
		vertices.clear();
		for (int i = 0; i < OneRingFaces.size(); i++) {
			int fi = OneRingFaces[i];
			int P0 = F(fi, 0);
			int P1 = F(fi, 1);
			int P2 = F(fi, 2);

			//check if the vertex already exist
			if (!(find(vertices.begin(), vertices.end(), P0) != vertices.end())) {
				vertices.push_back(P0);
			}
			if (!(find(vertices.begin(), vertices.end(), P1) != vertices.end())) {
				vertices.push_back(P1);
			}
			if (!(find(vertices.begin(), vertices.end(), P2) != vertices.end())) {
				vertices.push_back(P2);
			}
		}
		return vertices;
	}

	static std::vector<std::vector<int>> get_one_ring_vertices(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
		std::vector<std::vector<int> > VF, VFi;
		std::vector<std::vector<int>> OneRingVertices;
		OneRingVertices.resize(V.rows());
		igl::vertex_triangle_adjacency(V, F, VF, VFi);
		for (int vi = 0; vi < V.rows(); vi++) {
			std::vector<int> OneRingFaces = VF[vi];
			OneRingVertices[vi] = temp_get_one_ring_vertices_per_vertex(F, OneRingFaces);
		}
		return OneRingVertices;
	}

	static std::vector<std::vector<int>> get_adjacency_vertices_per_face(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
		std::vector<std::vector<int>> OneRingVertices = get_one_ring_vertices(V, F);
		std::vector<std::vector<int>> adjacency;
		adjacency.resize(F.rows());
		for (int fi = 0; fi < F.rows(); fi++) {
			adjacency[fi].clear();
			for (int vi = 0; vi < 3; vi++) {
				int Vi = F(fi, vi);
				std::vector<int> vi_oneRing = OneRingVertices[Vi];
				for (int v_index : vi_oneRing)
					if (!(find(adjacency[fi].begin(), adjacency[fi].end(), v_index) != adjacency[fi].end()))
						adjacency[fi].push_back(v_index);
			}
		}
		return adjacency;
	}

	static void center_of_mesh(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		Eigen::MatrixXd& center0,
		Eigen::VectorXd& radius0)
	{
		center0.resize(F.rows(), 3);
		radius0.resize(F.rows(), 1);
		Eigen::Vector3d avg;
		avg.setZero();
		for (int vi = 0; vi < V.rows(); vi++)
			avg += V.row(vi);
		avg /= V.rows();

		//update center0
		center0.col(0) = Eigen::VectorXd::Constant(F.rows(), avg(0));
		center0.col(1) = Eigen::VectorXd::Constant(F.rows(), avg(1));
		center0.col(2) = Eigen::VectorXd::Constant(F.rows(), avg(2));

		//update radius0
		for (int fi = 0; fi < F.rows(); fi++) {
			int x0 = F(fi, 0);
			int x1 = F(fi, 1);
			int x2 = F(fi, 2);
			Eigen::VectorXd x = (V.row(x0) + V.row(x1) + V.row(x2)) / 3;
			radius0(fi) = (x - avg).norm();
		}
	}

	static std::vector<std::set<int>> Triangle_triangle_adjacency(const Eigen::MatrixX3i& F) {
		std::vector<std::vector<std::vector<int>>> TT;
		igl::triangle_triangle_adjacency(F, TT);
		assert(TT.size() == F.rows());
		std::vector<std::set<int>> neigh; neigh.clear();

		for (int fi = 0; fi < TT.size(); fi++) {
			assert(TT[fi].size() == 3 && "Each face should be a triangle (not square for example)!");
			std::set<int> neigh_faces; neigh_faces.clear();
			neigh_faces.insert(fi);
			for (std::vector<int> hinge : TT[fi])
				for (int Face_neighbor : hinge)
					neigh_faces.insert(Face_neighbor);
			neigh.push_back(neigh_faces);
		}
		return neigh;
	}

	static Eigen::MatrixX3d Vertices_Neighbors(
		const int fi,
		const int distance,
		const Eigen::MatrixXd& V,
		const std::vector<std::set<int>>& TT,
		const std::vector<std::vector<int>>& TV)
	{
		std::set<int> faces;
		if (distance < 1) {
			std::cout << "Error! Distance should be 1 or Greater! (OptimizationUtils::Vertices_Neighbors)";
			exit(1);
		}
		else {
			faces = { fi };
			for (int i = 1; i < distance; i++) {
				std::set<int> currfaces = faces;
				for (int neighF : currfaces)
					faces.insert(TT[neighF].begin(), TT[neighF].end());
			}
		}

		std::set<int> neigh; neigh.clear();
		for (int currF : faces)
			for (int n : TV[currF])
				neigh.insert(n);
		
		Eigen::MatrixX3d neigh_vertices(neigh.size(), 3);
		int i = 0;
		for (int vi : neigh) {
			neigh_vertices.row(i++) = V.row(vi);
		}
		return neigh_vertices;
	}

	static double Least_Squares_Sphere_Fit_perFace(
		const int fi,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixX3d& vertices_indices,
		Eigen::MatrixXd& center0,
		Eigen::VectorXd& radius0)
	{
		//for more info:
		//https://jekel.me/2015/Least-Squares-Sphere-Fit/
		const int n = vertices_indices.rows();
		Eigen::MatrixXd A(n, 4);
		Eigen::VectorXd c(4), f(n);
		for (int ni = 0; ni < n; ni++) {
			const double xi = vertices_indices(ni, 0);
			const double yi = vertices_indices(ni, 1);
			const double zi = vertices_indices(ni, 2);
			A.row(ni) << 2 * xi, 2 * yi, 2 * zi, 1;
			f(ni) = pow(xi, 2) + pow(yi, 2) + pow(zi, 2);
		}
		//solve Ac = f and get c!
		c = (A.transpose() * A).colPivHouseholderQr().solve(A.transpose() * f);
		//after we got the solution c we pick from c: radius & center=(X,Y,Z)
		center0.row(fi) << c(0), c(1), c(2);
		radius0(fi) = sqrt(c(3) + pow(c(0), 2) + pow(c(1), 2) + pow(c(2), 2));
		//calculate MSE
		double toatal_MSE = 0;
		for (int ni = 0; ni < n; ni++)
			toatal_MSE += pow((vertices_indices.row(ni) - center0.row(fi)).squaredNorm() - pow(radius0(fi), 2), 2);
		toatal_MSE /= n;
		return toatal_MSE;
	}

	static void Least_Squares_Sphere_Fit(
		const int Distance_from,
		const int Distance_to,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		Eigen::MatrixXd& center0,
		Eigen::VectorXd& radius0)
	{
		//for more info:
		//https://jekel.me/2015/Least-Squares-Sphere-Fit/
		center0.resize(F.rows(), 3);
		radius0.resize(F.rows(), 1);
		std::vector<std::vector<int>> TV = get_adjacency_vertices_per_face(V, F);
		std::vector<std::set<int>> TT = Triangle_triangle_adjacency(F);
		
		for (int fi = 0; fi < F.rows(); fi++) {
			double minMSE = std::numeric_limits<double>::infinity();
			int argmin = -1;
			for (int d = Distance_from; d <= Distance_to; d++) {
				double currMSE = Least_Squares_Sphere_Fit_perFace(fi, V, F,
					Vertices_Neighbors(fi, d,V, TT, TV),
					center0, radius0);
				if (currMSE < minMSE) {
					minMSE = currMSE;
					argmin = d;
				}
			}
			std::cout << "fi =\t" << fi << "\t, argmin = " << argmin<< "\t, minMSE = " << minMSE << std::endl;
			Least_Squares_Sphere_Fit_perFace(fi, V, F,
				Vertices_Neighbors(fi, argmin, V, TT, TV),
				center0, radius0);
		}
	}
	
	static Eigen::MatrixXd center_per_triangle(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F)
	{
		Eigen::MatrixXd centers(F.rows(), 3);
		for (int fi = 0; fi < F.rows(); fi++) {
			int x0 = F(fi, 0);
			int x1 = F(fi, 1);
			int x2 = F(fi, 2);
			centers.row(fi) = (V.row(x0) + V.row(x1) + V.row(x2)) / 3;
		}
		return centers;
	}

	class Timer {
	private:
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		std::chrono::duration<double> duration;
		double* sum;
		double* curr;
	public:
		Timer(double* sum, double* current) {
			start = std::chrono::high_resolution_clock::now();
			this->sum = sum;
			this->curr = current;
		}
		~Timer() {
			end = std::chrono::high_resolution_clock::now();
			duration = end - start;
			double ms = duration.count() * 1000.0f;
			*sum += ms;
			*curr = ms;
			//std::cout << "Timer took " << ms << "ms\n";
		}
	};
}
