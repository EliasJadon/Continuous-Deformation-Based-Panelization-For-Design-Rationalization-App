#pragma once
#include "Utils/GUI.h"

namespace GUIExtensions {
	class MeshSimplificationData {
	public:
		Eigen::MatrixXd center_of_faces, center_of_sphere, normals;
		Clustering_Colors clustering_colors;
		Eigen::VectorXd radiuses;
		std::vector<std::vector<int>> clustering_faces_indices;
		std::vector<std::pair<int, int>> SelfIntersection_pairs, flippedFaces_pairs;
		Eigen::MatrixX3d clustering_faces_colors;
		std::shared_ptr <ObjectiveFunctions::Panels::AuxSphere> Energy_auxSphere;
		std::shared_ptr <ObjectiveFunctions::Panels::AuxPlanar> Energy_auxPlanar;
		std::shared_ptr <BendingNormal> Energy_Planar;
		std::shared_ptr <ObjectiveFunctions::Deformation::PinChosenVertices> Energy_PinChosenVertices;
		std::shared_ptr<NumericalOptimizations::Basic> minimizer;
		std::shared_ptr<TotalObjective> totalObjective;
		float prev_camera_zoom;
		Eigen::Vector3f prev_camera_translation;
		Eigen::Quaternionf prev_trackball_angle;
		Eigen::MatrixXd color_per_face, color_per_sphere_center, color_per_vertex_center;
		Eigen::MatrixXd color_per_face_norm, color_per_sphere_edge, color_per_norm_edge;
		int ModelID, CoreID;
		ImVec2 screen_position, screen_size, results_window_position, outputs_window_position;
		bool showSphereEdges, showNormEdges, showTriangleCenters, showSphereCenters, showFacesNorm;



		MeshSimplificationData(
			const int CoreID,
			const int meshID,
			igl::opengl::glfw::Viewer* viewer);
		~MeshSimplificationData() = default;
		double getRadiusOfSphere(int index);
		Eigen::VectorXd getRadiusOfSphere();
		Eigen::MatrixXd getCenterOfFaces();
		Eigen::MatrixXd getFacesNormals();
		Eigen::MatrixXd getFacesNorm();
		std::vector<int> GlobNeighSphereCenters(const int fi, const float distance);
		std::vector<int> FaceNeigh(const Eigen::Vector3d center, const float distance);
		std::vector<int> GlobNeighNorms(const int fi, const float distance);
		std::vector<int> getNeigh(const app_utils::Neighbor_Type type, const Eigen::MatrixXi& F, const int fi, const float distance);
		std::vector<int> adjSetOfTriangles(const Eigen::MatrixXi& F, const std::vector<int> selected, std::vector<std::vector<std::vector<int>>> TT);
		std::vector<int> vectorsIntersection(const std::vector<int>& A, const std::vector<int>& B);
		Eigen::MatrixXd getCenterOfSphere();
		Eigen::MatrixXd getSphereEdges();
		Eigen::MatrixX4d getValues(const app_utils::Face_Colors face_coloring_Type);
		void initFaceColors(
			const int numF,
			const Eigen::Vector3f center_sphere_color,
			const Eigen::Vector3f center_vertex_color,
			const Eigen::Vector3f centers_sphere_edge_color,
			const Eigen::Vector3f centers_norm_edge_color,
			const Eigen::Vector3f face_norm_color);
		void setFaceColors(const int fi, const Eigen::Vector3d color);
		void shiftFaceColors(const int fi, const double alpha, const Eigen::Vector3f model_color, const Eigen::Vector3f color);
		void initMinimizers(
			const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
			const OptimizationUtils::InitSphereAuxVariables& typeAuxVar,
			const int distance_from, const int distance_to,
			const double minus_normals_radius_length);
	};
};
