#include "GUIExtensions/MeshSimplificationData.h"
#include "NumericalOptimizations/InitAuxVar.h"

using namespace GUIExtensions;

MeshSimplificationData::MeshSimplificationData(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	const int CoreID,
	const int meshID,
	igl::opengl::glfw::Viewer* viewer)
{
	this->CoreID = CoreID;
	this->ModelID = meshID;
	showFacesNorm = showSphereEdges = showNormEdges = showTriangleCenters = showSphereCenters = false;
	showCylinderDir = false;
	this->center_of_faces = OptimizationUtils::center_per_triangle(V, F);
}

std::vector<int> MeshSimplificationData::GlobNeighCylinders(const int fi, const float distance) 
{
	std::vector<int> Neighbors; Neighbors.clear();
	for (int i = 0; i < C.rows(); i++)
		if (((C.row(fi) - C.row(i)).norm() + (A.row(fi) - A.row(i)).norm() + pow(R(fi) - R(i),2)) < distance)
			Neighbors.push_back(i);
	return Neighbors;
}

std::vector<int> MeshSimplificationData::GlobNeighSphereCenters(const int fi, const float distance) 
{
	std::vector<int> Neighbors; Neighbors.clear();
	for (int i = 0; i < C.rows(); i++)
		if (((C.row(fi) - C.row(i)).norm() + pow(R(fi) - R(i),2)) < distance)
			Neighbors.push_back(i);
	return Neighbors;
}

std::vector<int> MeshSimplificationData::FaceNeigh(const Eigen::Vector3d center, const float distance) 
{
	std::vector<int> Neighbors; Neighbors.clear();
	for (int i = 0; i < center_of_faces.rows(); i++) {
		double x = center(0) - center_of_faces(i, 0);
		double y = center(1) - center_of_faces(i, 1);
		double z = center(2) - center_of_faces(i, 2);
		if ((pow(x, 2) + pow(y, 2) + pow(z, 2)) < pow(distance,2))
			Neighbors.push_back(i);
	}
	return Neighbors;
}

std::vector<int> MeshSimplificationData::GlobNeighNorms(const int fi, const float distance) 
{
	std::vector<int> Neighbors; Neighbors.clear();
	for (int i = 0; i < N.rows(); i++)
		if ((N.row(fi) - N.row(i)).squaredNorm() < distance)
			Neighbors.push_back(i);
	return Neighbors;
}

std::vector<int> MeshSimplificationData::getNeigh(
	const app_utils::Neighbor_Type type, 
	const Eigen::MatrixXi& F, 
	const int fi, 
	const float distance) 
{
	std::vector<int> neigh;
	if (type == app_utils::Neighbor_Type::CURR_FACE)
		return { fi };
	if (type == app_utils::Neighbor_Type::GLOBAL_NORMALS)
		return GlobNeighNorms(fi, distance);
	if (type == app_utils::Neighbor_Type::GLOBAL_SPHERE)
		return GlobNeighSphereCenters(fi, distance);
	if (type == app_utils::Neighbor_Type::GLOBAL_CYLINDERS)
		return GlobNeighCylinders(fi, distance);
	if (type == app_utils::Neighbor_Type::LOCAL_NORMALS)
		neigh = GlobNeighNorms(fi, distance);
	if (type == app_utils::Neighbor_Type::LOCAL_SPHERE)
		neigh = GlobNeighSphereCenters(fi, distance);
	if (type == app_utils::Neighbor_Type::LOCAL_CYLINDERS)
		neigh = GlobNeighCylinders(fi, distance);
	
	//pick only adjanced faces in order to get local faces
	std::vector<int> result; result.push_back(fi);
	std::vector<std::vector<std::vector<int>>> TT;
	igl::triangle_triangle_adjacency(F, TT);
	int prevSize;
	do {
		prevSize = result.size();
		result = vectorsIntersection(adjSetOfTriangles(F, result, TT), neigh);
	} while (prevSize != result.size());
	return result;
}

std::vector<int> MeshSimplificationData::adjSetOfTriangles(
	const Eigen::MatrixXi& F, 
	const std::vector<int> selected, 
	std::vector<std::vector<std::vector<int>>> TT) 
{
	std::vector<int> adj = selected;
	for (int selectedFace : selected) {
		for (std::vector<int> _ : TT[selectedFace]) {
			for (int fi : _) {
				if (std::find(adj.begin(), adj.end(), fi) == adj.end())
					adj.push_back(fi);
			}
		}
	}
	return adj;
}

std::vector<int> MeshSimplificationData::vectorsIntersection(
	const std::vector<int>& A, 
	const std::vector<int>& B) 
{
	std::vector<int> intersection;
	for (int fi : A) {
		if (std::find(B.begin(), B.end(), fi) != B.end())
			intersection.push_back(fi);
	}
	return intersection;
}

void MeshSimplificationData::initFaceColors(
	const int numF,
	const Eigen::Vector3f center_sphere_color,
	const Eigen::Vector3f center_vertex_color,
	const Eigen::Vector3f centers_sphere_edge_color,
	const Eigen::Vector3f centers_norm_edge_color,
	const Eigen::Vector3f face_norm_color)
{
	color_per_face.resize(numF, 3);
	color_per_sphere_center.resize(numF, 3);
	color_per_vertex_center.resize(numF, 3);
	color_per_face_norm.resize(numF, 3);
	color_per_sphere_edge.resize(numF, 3);
	color_per_norm_edge.resize(numF, 3);
	for (int fi = 0; fi < numF; fi++) {
		color_per_sphere_center.row(fi) = center_sphere_color.cast<double>();
		color_per_vertex_center.row(fi) = center_vertex_color.cast<double>();
		color_per_face_norm.row(fi) = face_norm_color.cast<double>();
		color_per_sphere_edge.row(fi) = centers_sphere_edge_color.cast<double>();
		color_per_norm_edge.row(fi) = centers_norm_edge_color.cast<double>();
	}
}

void MeshSimplificationData::setFaceColors(const int fi, const Eigen::Vector3d color) {
	color_per_face.row(fi) = color;
	color_per_sphere_center.row(fi) = color;
	color_per_vertex_center.row(fi) = color;
	color_per_face_norm.row(fi) = color;
	color_per_sphere_edge.row(fi) = color;
	color_per_norm_edge.row(fi) = color;
}

void MeshSimplificationData::shiftFaceColors(
	const int fi, 
	const double alpha,
	const Eigen::Vector3f model_color,
	const Eigen::Vector3f color) 
{
	double w = std::min<double>(std::max<double>(alpha, 0), 1);
	auto averaged = color.cast<double>() * w + model_color.cast<double>() * (1 - w);
	color_per_face.row(fi) = averaged;
	color_per_sphere_center.row(fi) = averaged;
	color_per_vertex_center.row(fi) = averaged;
	color_per_face_norm.row(fi) = averaged;
	color_per_sphere_edge.row(fi) = averaged;
	color_per_norm_edge.row(fi) = averaged;
}

void MeshSimplificationData::initMinimizers(
	const Eigen::MatrixXd& V, 
	const Eigen::MatrixXi& F,
	const NumericalOptimizations::InitAuxVar::type& init_aux_var_type,
	const int NeighLevel, 
	const double manual_radius_value,
	const Eigen::RowVector3d manual_cylinder_dir,
	const Eigen::RowVector3d helper_vector_dir,
	const Eigen::MatrixXd& manual_A,
	const Eigen::VectorXd& manual_R)
{
	N.resize(F.rows(), 3);
	C.resize(F.rows(), 3);
	A.resize(F.rows(), 3);
	R.resize(F.rows());
	N.setZero();
	C.setZero();
	A.setZero();
	R.setZero();

	Eigen::MatrixX3d N_temp;
	igl::per_face_normals((Eigen::MatrixX3d)V, (Eigen::MatrixX3i)F, N_temp);
	N = N_temp;
	this->center_of_faces = OptimizationUtils::center_per_triangle(V, F);

	switch (init_aux_var_type) {
	
	case NumericalOptimizations::InitAuxVar::SPHERE_AUTO:
		NumericalOptimizations::InitAuxVar::sphere_fit_wrapper(NeighLevel, V, F, C, R);
		break;
	
	case NumericalOptimizations::InitAuxVar::SPHERE_AUTO_ALIGNED_TO_NORMAL:
		NumericalOptimizations::InitAuxVar::sphere_fit_aligned_to_normal_wrapper(NeighLevel, V, F, C, R);
		break;
	
	case NumericalOptimizations::InitAuxVar::SPHERE_MANUAL_ALIGNED_TO_NORMAL:
		R.setConstant(manual_radius_value);
		for (int i = 0; i < C.rows(); i++)
			C.row(i) = this->center_of_faces.row(i) - R(i) * N.row(i);
		break;
	
	case NumericalOptimizations::InitAuxVar::SPHERE_AUTO_CENTER_POINT:
		OptimizationUtils::center_of_mesh(V, F, C, R);
		break;
	
	case NumericalOptimizations::InitAuxVar::CYLINDER_AUTO:
		NumericalOptimizations::InitAuxVar::cylinder_fit_wrapper(18, 18, NeighLevel, V, F, C, A, R);
		break;
	
	case NumericalOptimizations::InitAuxVar::CYLINDER_AUTO_ALIGNED_TO_NORMAL:
		//TODO: implement the function
		NumericalOptimizations::InitAuxVar::cylinder_fit_wrapper(18, 18, NeighLevel, V, F, C, A, R);
		R.setConstant(manual_radius_value);
		for (int i = 0; i < C.rows(); i++)
			C.row(i) = this->center_of_faces.row(i) - R(i) * N.row(i);
		break;
	
	case NumericalOptimizations::InitAuxVar::CYLINDER_MANUAL_ALIGNED_TO_NORMAL:
		R.setConstant(manual_radius_value);
		for (int i = 0; i < C.rows(); i++)
			C.row(i) = this->center_of_faces.row(i) - R(i) * N.row(i);
		for (int fi = 0; fi < F.rows(); fi++)
			A.row(fi) = manual_cylinder_dir.normalized();
		break;

	case NumericalOptimizations::InitAuxVar::CYLINDER_MANUAL_PER_FACE_ALIGNED_TO_NORMAL:
		for (int fi = 0; fi < F.rows(); fi++) {
			A.row(fi) = manual_A.row(fi).normalized();
			R(fi) = manual_R(fi);
		}
		for (int i = 0; i < C.rows(); i++)
			C.row(i) = this->center_of_faces.row(i) - R(i) * N.row(i);	
		break;

	case NumericalOptimizations::InitAuxVar::CYLINDER_VECTOR_HELPER_ALIGNED_TO_NORMAL:
		R.setConstant(manual_radius_value);
		for (int i = 0; i < C.rows(); i++)
			C.row(i) = this->center_of_faces.row(i) - R(i) * N.row(i);
		for (int fi = 0; fi < F.rows(); fi++) {
			Eigen::RowVector3d N(N.row(fi));
			Eigen::RowVector3d H(helper_vector_dir.normalized());
			Eigen::RowVector3d B1(N.cross(H).normalized());
			Eigen::RowVector3d B2(N.cross(B1).normalized());
			if (B2.norm() == 0)
				A.row(fi) = helper_vector_dir.normalized();
			else
				A.row(fi) = B2;
		}
		break;
	}
	
	/*for (int ai = 1; ai < A.rows(); ai++) {
		double a = (A.row(ai) - A.row(0)).squaredNorm();
		double b = (A.row(ai) + A.row(0)).squaredNorm();
		if (b < a) {
			A.row(ai) = -A.row(ai);
		}
	}*/


	minimizer->init(V, N, C, R, A);
}

Eigen::MatrixX4d MeshSimplificationData::getValues(const app_utils::Face_Colors face_coloring_Type) {
	Eigen::MatrixX4d values(N.rows(), 4);
	for (int fi = 0; fi < N.rows(); fi++) {
		values.row(fi) = Eigen::Vector4d(N(fi, 0), N(fi, 1), N(fi, 2), 0);
		if (face_coloring_Type == app_utils::Face_Colors::SPHERE)
			values.row(fi) = Eigen::Vector4d(C(fi, 0), C(fi, 1), C(fi, 2), R[fi]);
	}
	return values;
}

