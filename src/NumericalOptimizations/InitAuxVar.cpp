#include "NumericalOptimizations/InitAuxVar.h"
#include <igl/per_face_normals.h>

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

static Eigen::MatrixX3d get_adjacent_vertices_per_face(
	const int fi,
	const int adjacency_level,
	const Eigen::MatrixXd& V,
	const std::vector<std::set<int>>& TT,
	const std::vector<std::vector<int>>& TV)
{
	std::set<int> faces;
	if (adjacency_level < 1) {
		std::cout << "Error! adjacency_level should be 1 or Greater!";
		exit(1);
	}
	else {
		faces = { fi };
		for (int i = 1; i < adjacency_level; i++) {
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

static Eigen::RowVector4d sphere_fit(const Eigen::MatrixX3d& point_cloud) {
	//for more info:
	//https://jekel.me/2015/Least-Squares-Sphere-Fit/
	
	Eigen::MatrixXd A(point_cloud.rows(), 4);
	Eigen::VectorXd c(4), f(point_cloud.rows());
	for (int ni = 0; ni < point_cloud.rows(); ni++) {
		const double xi = point_cloud(ni, 0);
		const double yi = point_cloud(ni, 1);
		const double zi = point_cloud(ni, 2);
		A.row(ni) << 2 * xi, 2 * yi, 2 * zi, 1;
		f(ni) = pow(xi, 2) + pow(yi, 2) + pow(zi, 2);
	}
	//solve Ac = f and get c!
	c = (A.transpose() * A).colPivHouseholderQr().solve(A.transpose() * f);
	return Eigen::RowVector4d(c(0), c(1), c(2), sqrt(c(3) + pow(c(0), 2) + pow(c(1), 2) + pow(c(2), 2)));
}

static Eigen::Vector4d sphere_fit_aligned_to_normal(
	const Eigen::RowVector3d& face_center_point,
	const Eigen::RowVector3d& face_normal,
	const Eigen::MatrixX3d& point_cloud)
{
	Eigen::MatrixXd A(point_cloud.rows(), 1);
	Eigen::VectorXd r(1), f(point_cloud.rows());
	for (int pi = 0; pi < point_cloud.rows(); pi++) {
		A(pi, 0) = (point_cloud.row(pi) - face_center_point).squaredNorm();
		f(pi) = 2 * face_normal.dot(point_cloud.row(pi) - face_center_point);
	}
	//solve Ac = f and get c!
	r = (A.transpose() * A).colPivHouseholderQr().solve(A.transpose() * f);
	return Eigen::Vector4d(
		face_center_point(0) + r(0) * face_normal(0),
		face_center_point(1) + r(1) * face_normal(1),
		face_center_point(2) + r(2) * face_normal(2),
		r(0));
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


void NumericalOptimizations::InitAuxVar::sphere_fit_wrapper(
	const int adjacency_level,
	const Eigen::MatrixXd& V, 
	const Eigen::MatrixXi& F,
	Eigen::MatrixXd& C, 
	Eigen::VectorXd& R)
{
	C.resize(F.rows(), 3);
	R.resize(F.rows(), 1);
	std::vector<std::vector<int>> TV = get_adjacency_vertices_per_face(V, F);
	std::vector<std::set<int>> TT = Triangle_triangle_adjacency(F);

	for (int fi = 0; fi < F.rows(); fi++) {
		Eigen::MatrixX3d point_cloud = get_adjacent_vertices_per_face(fi, adjacency_level, V, TT, TV);
		Eigen::RowVector4d sphere = sphere_fit(point_cloud);
		C.row(fi) << sphere(0), sphere(1), sphere(2);
		R(fi) = sphere(3);
	}
}

void NumericalOptimizations::InitAuxVar::sphere_fit_aligned_to_normal_wrapper(
	const int adjacency_level,
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	Eigen::MatrixXd& C,
	Eigen::VectorXd& R)
{
	C.resize(F.rows(), 3);
	R.resize(F.rows(), 1);
	std::vector<std::vector<int>> TV = get_adjacency_vertices_per_face(V, F);
	std::vector<std::set<int>> TT = Triangle_triangle_adjacency(F);
	Eigen::MatrixX3d N;
	igl::per_face_normals(V, F, N);

	for (int fi = 0; fi < F.rows(); fi++) {
		Eigen::MatrixX3d point_cloud = get_adjacent_vertices_per_face(fi, adjacency_level, V, TT, TV);
		Eigen::RowVector3d face_center_point = (V.row(F(fi, 0)) + V.row(F(fi, 1)) + V.row(F(fi, 2))) / 3;
		Eigen::RowVector4d sphere = sphere_fit_aligned_to_normal(face_center_point, N.row(fi), point_cloud);
		C.row(fi) << sphere(0), sphere(1), sphere(2);
		R(fi) = sphere(3);
	}
}
