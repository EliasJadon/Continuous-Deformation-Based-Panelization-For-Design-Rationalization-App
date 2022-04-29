#include "NumericalOptimizations/InitAuxVar.h"

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


void NumericalOptimizations::InitAuxVar::general_sphere_fit(
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
				Vertices_Neighbors(fi, d, V, TT, TV),
				center0, radius0);
			if (currMSE < minMSE) {
				minMSE = currMSE;
				argmin = d;
			}
		}
		std::cout << "fi =\t" << fi << "\t, argmin = " << argmin << "\t, minMSE = " << minMSE << std::endl;
		Least_Squares_Sphere_Fit_perFace(fi, V, F,
			Vertices_Neighbors(fi, argmin, V, TT, TV),
			center0, radius0);
	}
}