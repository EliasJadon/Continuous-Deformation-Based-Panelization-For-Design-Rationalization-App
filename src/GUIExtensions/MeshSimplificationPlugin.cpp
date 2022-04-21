#include "MeshSimplificationPlugin.h"
#include <igl/file_dialog_open.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <igl/writeOFF.h>
#include <igl/boundary_loop.h>
#include <igl/readOFF.h>
#include <igl/adjacency_matrix.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>



#define ADDING_WEIGHT_PER_HINGE_VALUE 10.0f
#define MAX_WEIGHT_PER_HINGE_VALUE  500.0f //50.0f*ADDING_WEIGHT_PER_HINGE_VALUE
#define MAX_SIGMOID_PER_HINGE_VALUE  40.0f //50.0f*ADDING_WEIGHT_PER_HINGE_VALUE

using namespace GUIExtensions;

MeshSimplificationPlugin::MeshSimplificationPlugin(){
	
}

IGL_INLINE void MeshSimplificationPlugin::init(igl::opengl::glfw::Viewer *_viewer)
{
	ImGuiPlugin::init(_viewer);
	for (int i = 0; i < 7; i++)
		CollapsingHeader_prev[i] = CollapsingHeader_curr[i] = false;
	UserInterface_UpdateAllOutputs = false;
	CollapsingHeader_change = false;
	neighbor_distance = brush_radius = 0.3;
	initSphereAuxVariables = OptimizationUtils::InitSphereAuxVariables::MINUS_NORMALS;
	IsMouseDraggingAnyWindow = false;
	isMinimizerRunning = false;
	energies_window = results_window = outputs_window = true;
	neighbor_Type = app_utils::Neighbor_Type::CURR_FACE;
	isUpdateAll = true;
	face_coloring_Type = app_utils::Face_Colors::NO_COLORS;
	clustering_brightness_w = 0.65;
	faceColoring_type = 0;
	optimizer_type = Cuda::OptimizerType::Adam;
	linesearch_type = OptimizationUtils::LineSearch::FUNCTION_VALUE;
	view = app_utils::View::HORIZONTAL;
	Max_Distortion = 5;
	Vertex_Energy_color = RED_COLOR;
	Highlighted_face_color = Eigen::Vector3f(153 / 255.0f, 0, 153 / 255.0f);
	Neighbors_Highlighted_face_color = Eigen::Vector3f(1, 102 / 255.0f, 1);
	center_sphere_color = Eigen::Vector3f(0, 1, 1);
	center_vertex_color = Eigen::Vector3f(128 / 255.0f, 128 / 255.0f, 128 / 255.0f);
	Color_sphere_edges = Color_normal_edge = Eigen::Vector3f(0 / 255.0f, 100 / 255.0f, 100 / 255.0f);
	face_norm_color = Eigen::Vector3f(0, 1, 1);
	Fixed_vertex_color = BLUE_COLOR;
	Dragged_vertex_color = GREEN_COLOR;
	model_color = GREY_COLOR;
	text_color = BLACK_COLOR;
	isPluginInitialized = true;
	glfwMaximizeWindow(viewer->window);
	load_first_mesh(modelName, original_V, original_F);
}

void MeshSimplificationPlugin::load_first_mesh(const std::string& name, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
	modelName = name;
	original_V = V;
	original_F = F;
	
	assert((viewer->data().F.rows() == 0 && viewer->data().V.rows() == 0) && "Error! Invalid state\n");
	assert(viewer->data_list.size() == 1);
	assert(viewer->core_list.size() == 1);
	
	inputModelID = viewer->data_list[0].id;
	viewer->data(inputModelID).clear();
	viewer->data(inputModelID).set_mesh(V, F);
	viewer->data(inputModelID).compute_normals();
	viewer->data(inputModelID).uniform_colors(
		Eigen::Vector3d(51.0 / 255.0, 43.0 / 255.0, 33.3 / 255.0),
		Eigen::Vector3d(255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0),
		Eigen::Vector3d(255.0 / 255.0, 235.0 / 255.0, 80.0 / 255.0));
	inputCoreID = viewer->core_list[0].id;
	
	Outputs.clear();
	add_output();
	assert(viewer->data_list.size() == 2);
	assert(viewer->core_list.size() == 2);
	assert(Outputs.size() == 1);
}

void MeshSimplificationPlugin::CollapsingHeader_update()
{
	CollapsingHeader_change = false;
	int changed_index = NOT_FOUND;
	for (int i = 0; i < 9; i++)
	{
		if (CollapsingHeader_curr[i] && !CollapsingHeader_prev[i])
		{
			changed_index = i;
			CollapsingHeader_change = true;
		}
	}
	if (CollapsingHeader_change)
	{
		for (int i = 0; i < 9; i++)
			CollapsingHeader_prev[i] = CollapsingHeader_curr[i] = false;
		CollapsingHeader_prev[changed_index] = CollapsingHeader_curr[changed_index] = true;
	}
}

void MeshSimplificationPlugin::CollapsingHeader_colors()
{
	if (CollapsingHeader_change)
		ImGui::SetNextItemOpen(CollapsingHeader_curr[0]);
	if (ImGui::CollapsingHeader("colors"))
	{
		CollapsingHeader_curr[0] = true;
		ImGui::ColorEdit3("Highlighted face", Highlighted_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Center sphere", center_sphere_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Center vertex", center_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Sphere edge", Color_sphere_edges.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Normal edge", Color_normal_edge.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Face norm", face_norm_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Neighbors Highlighted face", Neighbors_Highlighted_face_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Fixed vertex", Fixed_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Dragged vertex", Dragged_vertex_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Model", model_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit3("Vertex Energy", Vertex_Energy_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
		ImGui::ColorEdit4("Text", text_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);
	}
}

void MeshSimplificationPlugin::CollapsingHeader_face_coloring()
{
	if (CollapsingHeader_change)
		ImGui::SetNextItemOpen(CollapsingHeader_curr[1]);
	if (ImGui::CollapsingHeader("Face coloring"))
	{
		CollapsingHeader_curr[1] = true;
		ImGui::Combo("type", (int *)(&faceColoring_type), app_utils::build_color_energies_list(Outputs[0].totalObjective));
		ImGui::PushItemWidth(80 * ((igl::opengl::glfw::imgui::ImGuiMenu*)widgets.front())->menu_scaling());
		ImGui::DragFloat("Max Distortion", &Max_Distortion, 0.05f, 0.01f, 10000.0f);
		ImGui::PopItemWidth();
	}
}

void MeshSimplificationPlugin::CollapsingHeader_screen()
{
	if (CollapsingHeader_change)
		ImGui::SetNextItemOpen(CollapsingHeader_curr[2]);
	if (ImGui::CollapsingHeader("Screen options"))
	{
		CollapsingHeader_curr[2] = true;
		if (ImGui::Combo("View type", (int *)(&view), app_utils::build_view_names_list(Outputs.size())))
		{
			int frameBufferWidth, frameBufferHeight;
			glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
			post_resize(frameBufferWidth, frameBufferHeight);
		}
		if (view == app_utils::View::HORIZONTAL ||
			view == app_utils::View::VERTICAL)
		{
			if (ImGui::SliderFloat("Core Size", &core_size, 0, 1.0 / Outputs.size(), std::to_string(core_size).c_str(), 1))
			{
				int frameBufferWidth, frameBufferHeight;
				glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
				post_resize(frameBufferWidth, frameBufferHeight);
			}
		}
	}
}

void MeshSimplificationPlugin::CollapsingHeader_user_interface()
{
	if (!ImGui::CollapsingHeader("User Interface"))
	{
		ImGui::Checkbox("Update UI together", &UserInterface_UpdateAllOutputs);
		ImGui::Combo("Neighbor type", (int *)(&neighbor_Type), "Curr Face\0Local Sphere\0Global Sphere\0Local Normals\0Global Normals\0\0");
		ImGui::DragFloat("Neighbors Distance", &neighbor_distance, 0.0005f, 0.00001f, 10000.0f,"%.5f");
		ImGui::DragFloat("Brush Radius", &brush_radius);
		if (ImGui::Button("Clear sellected faces & vertices"))
			clear_sellected_faces_and_vertices();
	}
}

void MeshSimplificationPlugin::CollapsingHeader_measures() {
	if (CollapsingHeader_change)
		ImGui::SetNextItemOpen(CollapsingHeader_curr[7]);
	if (ImGui::CollapsingHeader("Measures")) {
		CollapsingHeader_curr[7] = true;

		//self-intersection Measure
		if (ImGui::Checkbox("self-intersection", &isChecking_SelfIntersection) ||
			ImGui::DragFloat("self-intersection epsilon", &SelfIntersection_epsilon, 0.000001f, 0, 1, "%.7f")) {
			if (isChecking_SelfIntersection) {
				for (int oi = 0; oi < Outputs.size(); oi++) {
					const Eigen::MatrixXd V = OutputModel(oi).V;
					const Eigen::MatrixXi F = OutputModel(oi).F;
					Outputs[oi].SelfIntersection_pairs.clear();
					for (int f1 = 0; f1 < F.rows(); f1++) {
						for (int f2 = f1 + 1; f2 < F.rows(); f2++) {
							if (app_utils::doTrianglesIntersect(f1, f2, V, F, SelfIntersection_epsilon)) {
								//There is self-intersection!!!
								Outputs[oi].SelfIntersection_pairs.push_back(std::pair<int, int>(f1, f2));
								//std::cout << "Self-intersection found between: " << f1 << ", " << f2 << std::endl;
							}
						}
					}
				}
			}
			else {
				for (auto& o : Outputs)
					o.SelfIntersection_pairs.clear();
			}
		}
		for (int oi = 0; oi < Outputs.size(); oi++) {
			ImGui::Text((std::to_string(oi) + ": " + std::to_string(Outputs[oi].SelfIntersection_pairs.size())).c_str());
		}

		//flipped-faces Measure
		if (ImGui::Checkbox("flipped-faces", &isChecking_FlippedFaces) ||
			ImGui::DragFloat("Flipped-faces epsilon", &flippedFaces_epsilon, 0.000001f, 0, 1, "%.5f")) {
			if (isChecking_FlippedFaces) {
				for (int oi = 0; oi < Outputs.size(); oi++) {
					const Eigen::MatrixXd V = OutputModel(oi).V;
					const Eigen::MatrixXi F = OutputModel(oi).F;
					auto& AS = Outputs[oi].Energy_auxSphere;
					Outputs[oi].flippedFaces_pairs = app_utils::getFlippedFaces(V, F, AS->hinges_faceIndex, flippedFaces_epsilon);
				}
			}
			else {
				for (auto& o : Outputs)
					o.flippedFaces_pairs.clear();
			}
		}
		for (int oi = 0; oi < Outputs.size(); oi++) {
			ImGui::Text((std::to_string(oi) + ": " + std::to_string(Outputs[oi].flippedFaces_pairs.size())).c_str());
		}

		//Mesh Diameter Measure
		static double diameter_input = 0;
		static std::vector<double> diameter_output;
		if (ImGui::Button("Mesh Diameter")) {
			diameter_input = app_utils::mesh_diameter(InputModel().V);
			diameter_output.clear();
			for (int oi = 0; oi < Outputs.size(); oi++)
				diameter_output.push_back(app_utils::mesh_diameter(OutputModel(oi).V));
		}
		if (diameter_input)
			ImGui::Text(("input: " + std::to_string(diameter_input)).c_str());
		for (int oi = 0; oi < diameter_output.size(); oi++)
			ImGui::Text((std::to_string(oi) + ": " + std::to_string(diameter_output[oi])).c_str());

		//diagonal_of_bounding_box Measure
		static double diag_input = 0;
		static std::vector<double> diag_output;
		if (ImGui::Button("BoundingBox Diag.")) {
			diag_input = app_utils::diagonal_of_bounding_box(InputModel().V);
			diag_output.clear();
			for (int oi = 0; oi < Outputs.size(); oi++)
				diag_output.push_back(app_utils::diagonal_of_bounding_box(OutputModel(oi).V));
		}
		if (diag_input)
			ImGui::Text(("input: " + std::to_string(diag_input)).c_str());
		for (int oi = 0; oi < diag_output.size(); oi++)
			ImGui::Text((std::to_string(oi) + ": " + std::to_string(diag_output[oi])).c_str());

		//Hausdorff-distance Measure
		static std::vector<double> HausdorffDis;
		if (ImGui::Button("Hausdorff-distance")) {
			HausdorffDis.clear();
			for (int oi = 0; oi < Outputs.size(); oi++)
				HausdorffDis.push_back(app_utils::Hausdorff_distance(OutputModel(oi).V, InputModel().V));
		}
		for (int oi = 0; oi < HausdorffDis.size(); oi++) {
			ImGui::Text((std::to_string(oi) + ": " + std::to_string(HausdorffDis[oi])).c_str());
			if (diag_input)
				ImGui::Text(("Ratio: " + std::to_string((HausdorffDis[oi] / diag_output[oi]) * 100) + "%%").c_str());
		}

		// Base-units Measure
		static double max_all = -1, min_all = -1, avg_all = -1, std_all = -1;
		if (ImGui::Button("Base-units Error")) {
			for (int oi = 0; oi < Outputs.size(); oi++) {
				const Eigen::MatrixXd& V = OutputModel(oi).V;
				const Eigen::MatrixXi& F = OutputModel(oi).F;
				const Eigen::MatrixXd& C = Outputs[oi].getCenterOfSphere();
				const std::vector<std::vector<int>>& clusters = Outputs[oi].clustering_faces_indices;
				if (face_coloring_Type == app_utils::Face_Colors::SPHERES_CLUSTERING)
					app_utils::sphere_error_distribution(V, F, C, clusters, max_all, min_all, avg_all, std_all);
				if (face_coloring_Type == app_utils::Face_Colors::NORMALS_CLUSTERING)
					app_utils::planar_error_distribution(V, F, clusters, max_all, min_all, avg_all, std_all);
			}
		}
		
		if (max_all != -1) {
			ImGui::Text(("Max: " + std::to_string(max_all * 1000) + "m").c_str());
			ImGui::Text(("Min: " + std::to_string(min_all * 1000) + "m").c_str());
			ImGui::Text(("Avg: " + std::to_string(avg_all * 1000) + "m").c_str());
			ImGui::Text(("Std: " + std::to_string(std_all * 1000) + "m").c_str());
		}

		//1-cluster Base-unit measure
		static bool is_Base_units_1_cluster = false;
		ImGui::Checkbox("Base-units-1-clus Err", &is_Base_units_1_cluster);
		if (is_Base_units_1_cluster) {
			for (int oi = 0; oi < Outputs.size(); oi++) {
				const Eigen::MatrixXd& V = OutputModel(oi).V;
				const Eigen::MatrixXi& F = OutputModel(oi).F;
				const Eigen::MatrixXd& C = Outputs[oi].getCenterOfSphere();
				const std::vector<std::vector<int>>& clusters = Outputs[oi].clustering_faces_indices;
				static double max = -1, min = -1, avg = -1, std = -1;
				int output_index, face_index;
				Eigen::Vector3f intersec_point;
				if (pick_face(output_index, face_index, intersec_point)) {
					int cluster_index = NOT_FOUND;
					//find the cluster index
					for (int ci = 0; ci < clusters.size(); ci++) {
						const std::vector<int>& currClus = clusters[ci];
						if (std::find(currClus.begin(), currClus.end(), face_index) != currClus.end())
							cluster_index = ci;
					}
					if (cluster_index != NOT_FOUND) {
						//output the cluster data
						ImGui::Text(("Cluster: " + std::to_string(cluster_index)).c_str());
						if (face_coloring_Type == app_utils::Face_Colors::SPHERES_CLUSTERING)
							app_utils::sphere_error_distribution(V, F, C, { clusters[cluster_index] }, max, min, avg, std);
						if (face_coloring_Type == app_utils::Face_Colors::NORMALS_CLUSTERING)
							app_utils::planar_error_distribution(V, F, { clusters[cluster_index] }, max, min, avg, std);
						if (max != -1) {
							ImGui::Text(("Max: " + std::to_string(max * 1000) + "m").c_str());
							ImGui::Text(("Min: " + std::to_string(min * 1000) + "m").c_str());
							ImGui::Text(("Avg: " + std::to_string(avg * 1000) + "m").c_str());
							ImGui::Text(("Std: " + std::to_string(std * 1000) + "m").c_str());
						}
					}
				}
			}
		}
	}
}

void MeshSimplificationPlugin::CollapsingHeader_fabrication() {
	if (CollapsingHeader_change)
		ImGui::SetNextItemOpen(CollapsingHeader_curr[8]);
	if (ImGui::CollapsingHeader("Fabrication"))
	{
		CollapsingHeader_curr[8] = true;
		static float scale = 1;
		ImGui::DragFloat("Fabrication scale", &scale, 0.001f, 0, 1);

		//print polygons for fabrication
		if (ImGui::Button("print polygons")) {
			std::cout << "Visit https://www.math10.com/en/geometry/geogebra/geogebra.html \n";
			for (int oi = 0; oi < Outputs.size(); oi++) {
				std::cout << "********\tThis is result (" << oi << ")\t********" << std::endl;
				for(int pi=0;pi< Outputs[oi].clustering_faces_indices.size();pi++){
					std::vector<int> clusfaces = Outputs[oi].clustering_faces_indices[pi];
					std::cout << "\n\n>> Polygon " << pi <<":\nP"<<pi<<"=";
					app_utils::getPlanarCoordinates(OutputModel(oi).V, OutputModel(oi).F, clusfaces, scale);
				}
			}
		}

	}
}

void MeshSimplificationPlugin::CollapsingHeader_clustering()
{
	if (CollapsingHeader_change)
		ImGui::SetNextItemOpen(CollapsingHeader_curr[3]);
	if (ImGui::CollapsingHeader("Clustering"))
	{
		CollapsingHeader_curr[3] = true;
		ImGui::Combo("Face Colors Type", (int*)(&face_coloring_Type), "No Colors\0Normals Clustering\0Spheres Clustering\0Sigmoid Parameter\0\0");
		ImGui::DragFloat("Bright. Weight", &clustering_brightness_w, 0.001f, 0, 1);
		if (ImGui::Combo("Clus. type", (int*)(&clusteringType), "None\0Circle Clus.\0Agglomerative hierarchical\0External Clus.\0")
			|| ImGui::DragFloat("MSE threshold", &Clustering_MSE_Threshold, 0.000001f, 0, 1, "%.8f")) {
			if (clusteringType == app_utils::ClusteringType::Circle_clustering) {
				for (auto& o : Outputs) {
					RadiusClustering(
						o.getValues(face_coloring_Type), 
						Clustering_MSE_Threshold, 
						o.clustering_faces_indices);
				}
			}
			if (clusteringType == app_utils::ClusteringType::Agglomerative_hierarchical) {
				for (auto& o : Outputs) {
					Agglomerative_hierarchical_clustering(
						o.getValues(face_coloring_Type),
						Clustering_MSE_Threshold,
						InputModel().F.rows(),
						o.clustering_faces_indices);
				}
			}
		}

		if (ImGui::Button("Change Clusters Colors")) {
			for (auto& o : Outputs)
				o.clustering_colors.changeColors();
		}

		static std::vector<bool> Check_clusters_separation;
		if (ImGui::Button("Check clusters separation")) {
			Check_clusters_separation.clear();
			for (int oi = 0; oi < Outputs.size(); oi++)
				Check_clusters_separation.push_back(
					app_utils::cluster_separation(
						OutputModel(oi).F,
						Outputs[oi].clustering_faces_indices)
				);
		}
		for (bool b : Check_clusters_separation) {
			ImGui::Text((std::to_string(b)).c_str());
		}
		
		
		//upload off file
		if (ImGui::Button("upload OFF file")) {
			stop_all_minimizers_threads();
			Eigen::MatrixXd V;
			Eigen::MatrixXi F;
			std::string uploadFilePath = igl::file_dialog_open();
			igl::readOFF(uploadFilePath, V, F);
			for (int i = 0; i < Outputs.size(); i++)
				Outputs[i].minimizer->upload_x(Eigen::Map<const Eigen::VectorXd>(V.data(), V.size()));
		}

		//Upload PLY2 in order to compare our work with others
		if (ImGui::Button("upload PLY2 file")) {
			Eigen::MatrixXd V;
			Eigen::MatrixX3i F;
			Eigen::VectorXi Clus;
			std::string uploadFilePath = igl::file_dialog_open();
			app_utils::readPLY2(uploadFilePath, V, F, Clus);
			
			if (InputModel().V.size() == V.size() && InputModel().F.size() == F.size()) {
				clusteringType = app_utils::ClusteringType::External;
				for (auto& o : Outputs)
					o.clustering_faces_indices = app_utils::get_Clusters_Vector_From_Matrix(Clus);
			}	
		}
	}
}

void MeshSimplificationPlugin::CollapsingHeader_minimizer()
{
	if (CollapsingHeader_change)
		ImGui::SetNextItemOpen(CollapsingHeader_curr[4]);
	if (ImGui::CollapsingHeader("Minimizer"))
	{
		CollapsingHeader_curr[4] = true;
		if (ImGui::Button("Run one iter"))
			run_one_minimizer_iter();
		if (ImGui::Checkbox("Run Minimizer", &isMinimizerRunning))
			isMinimizerRunning ? start_all_minimizers_threads() : stop_all_minimizers_threads();
		if (ImGui::Combo("Optimizer", (int *)(&optimizer_type), "Gradient Descent\0Adam\0\0"))
			change_minimizer_type(optimizer_type);
		if (ImGui::Combo("init sphere var", (int *)(&initSphereAuxVariables), "Sphere Fit\0Mesh Center\0Minus Normal\0\0"))
			init_aux_variables();
		if (initSphereAuxVariables == OptimizationUtils::InitSphereAuxVariables::MINUS_NORMALS &&
			ImGui::DragFloat("radius length", &radius_length_minus_normal, 0.01f, 0.0f, 1000.0f, "%.7f"))
			init_aux_variables();
		if (initSphereAuxVariables == OptimizationUtils::InitSphereAuxVariables::SPHERE_FIT) 
		{
			if (ImGui::DragInt("Neigh From", &(InitMinimizer_NeighLevel_From), 1, 1, 200))
				init_aux_variables();
			if (ImGui::DragInt("Neigh To", &(InitMinimizer_NeighLevel_To), 1, 1, 200))
				init_aux_variables();
		}

		if (ImGui::Combo("line search", (int *)(&linesearch_type), "Gradient Norm\0Function Value\0Constant Step\0\0")) {
			for (auto& o : Outputs)
				o.minimizer->lineSearch_type = linesearch_type;
		}
		if (linesearch_type == OptimizationUtils::LineSearch::CONSTANT_STEP && ImGui::DragFloat("Step value", &constantStep_LineSearch, 0.0001f, 0.0f, 1.0f, "%.7f")) {
			for (auto& o : Outputs)
				o.minimizer->constantStep_LineSearch = constantStep_LineSearch;	
		}
		if (ImGui::Button("Check gradients"))
			checkGradients();
	}
}

void MeshSimplificationPlugin::CollapsingHeader_cores(igl::opengl::ViewerCore& core, igl::opengl::ViewerData& data)
{
	if (!outputs_window)
		return;
	ImGui::PushID(core.id);
	if (CollapsingHeader_change)
		ImGui::SetNextItemOpen(CollapsingHeader_curr[5]);
	if (ImGui::CollapsingHeader(("Core " + std::to_string(data.id)).c_str()))
	{
		CollapsingHeader_curr[5] = true;
		if (ImGui::Button("Center object", ImVec2(-1, 0)))
			core.align_camera_center(data.V, data.F);
		if (ImGui::Button("Snap canonical view", ImVec2(-1, 0)))
			viewer->snap_to_canonical_quaternion();
		// Zoom & Lightining factor
		ImGui::PushItemWidth(80 * ((igl::opengl::glfw::imgui::ImGuiMenu*)widgets.front())->menu_scaling());
		ImGui::DragFloat("Zoom", &(core.camera_zoom), 0.05f, 0.1f, 100000.0f);
		ImGui::DragFloat("Lighting factor", &(core.lighting_factor), 0.05f, 0.1f, 20.0f);
		// Select rotation type
		int rotation_type = static_cast<int>(core.rotation_type);
		static Eigen::Quaternionf trackball_angle = Eigen::Quaternionf::Identity();
		static bool orthographic = true;
		if (ImGui::Combo("Camera Type", &rotation_type, "Trackball\0Two Axes\02D Mode\0\0"))
		{
			using RT = igl::opengl::ViewerCore::RotationType;
			auto new_type = static_cast<RT>(rotation_type);
			if (new_type != core.rotation_type)
			{
				if (new_type == RT::ROTATION_TYPE_NO_ROTATION)
				{
					trackball_angle = core.trackball_angle;
					orthographic = core.orthographic;
					core.trackball_angle = Eigen::Quaternionf::Identity();
					core.orthographic = true;
				}
				else if (core.rotation_type == RT::ROTATION_TYPE_NO_ROTATION)
				{
					core.trackball_angle = trackball_angle;
					core.orthographic = orthographic;
				}
				core.set_rotation_type(new_type);
			}
		}
		if(ImGui::Checkbox("Orthographic view", &(core.orthographic)) && isUpdateAll)
			for (auto& c : viewer->core_list)
				c.orthographic = core.orthographic;
		ImGui::PopItemWidth();
		if (ImGui::ColorEdit4("Background", core.background_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) && isUpdateAll)
			for (auto& c : viewer->core_list)
				c.background_color = core.background_color;
	}
	ImGui::PopID();
}

void MeshSimplificationPlugin::CollapsingHeader_models(igl::opengl::ViewerData& data)
{
	if (!outputs_window)
		return;
	auto make_checkbox = [&](const char *label, unsigned int &option) {
		bool temp = option;
		bool res = ImGui::Checkbox(label, &temp);
		option = temp;
		return res;
	};
	ImGui::PushID(data.id);
	if (CollapsingHeader_change)
		ImGui::SetNextItemOpen(CollapsingHeader_curr[6]);
	if (ImGui::CollapsingHeader((modelName + " " + std::to_string(data.id)).c_str()))
	{
		CollapsingHeader_curr[6] = true;
		if (ImGui::Checkbox("Face-based", &(data.face_based)))
		{
			data.dirty = igl::opengl::MeshGL::DIRTY_ALL;
			if(isUpdateAll)
			{
				for (auto& d : viewer->data_list)
				{
					d.dirty = igl::opengl::MeshGL::DIRTY_ALL;
					d.face_based = data.face_based;
				}
			}
		}
		if (make_checkbox("Show texture", data.show_texture) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_texture = data.show_texture;
		if (ImGui::Checkbox("Invert normals", &(data.invert_normals))) {
			if (isUpdateAll)
			{
				for (auto& d : viewer->data_list)
				{
					d.dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
					d.invert_normals = data.invert_normals;
				}
			}
			else
				data.dirty |= igl::opengl::MeshGL::DIRTY_NORMAL;
		}
		if (make_checkbox("Show overlay", data.show_overlay) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_overlay = data.show_overlay;
		if (make_checkbox("Show overlay depth", data.show_overlay_depth) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_overlay_depth = data.show_overlay_depth;
		if (ImGui::ColorEdit4("Line color", data.line_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.line_color = data.line_color;
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
		if (ImGui::DragFloat("Shininess", &(data.shininess), 0.05f, 0.0f, 100.0f) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.shininess = data.shininess;
		ImGui::PopItemWidth();
		if (make_checkbox("Wireframe", data.show_lines) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_lines = data.show_lines;
		if (make_checkbox("Fill", data.show_faces) && isUpdateAll)
			for(auto& d: viewer->data_list)
				d.show_faces = data.show_faces;
		/*if (ImGui::Checkbox("Show vertex labels", &(data.show_vertex_labels)) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_vertex_labels = data.show_vertex_labels;
		if (ImGui::Checkbox("Show faces labels", &(data.show_face_labels)) && isUpdateAll)
			for (auto& d : viewer->data_list)
				d.show_face_labels = data.show_face_labels;*/
	}
	ImGui::PopID();
}

void MeshSimplificationPlugin::Draw_energies_window()
{
	if (!energies_window)
		return;
	ImGui::SetNextWindowPos(energies_window_position);
	//ImGui::SetNextWindowSize(ImVec2(20, 30));
	ImGui::Begin("Energies & Timing", NULL, ImGuiWindowFlags_AlwaysAutoResize);
	int id = 0;
	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
	if (ImGui::Button(("Add one more " + modelName).c_str()))
		add_output();
	ImGui::PopStyleColor();

	//add automatic lambda change
	if (ImGui::BeginTable("Lambda table", 12, ImGuiTableFlags_Resizable))
	{
		ImGui::TableNextRow();
		ImGui::TableNextColumn();
		ImGui::Text("ID");
		ImGui::TableNextColumn();
		ImGui::Text("Max Update");
		ImGui::TableNextColumn();
		ImGui::Text("On/Off");
		ImGui::TableNextColumn(); 
		ImGui::Text("Start from");
		ImGui::TableNextColumn(); 
		ImGui::Text("Stop at");
		ImGui::TableNextColumn(); 
		ImGui::Text("#iter//lambda");
		ImGui::TableNextColumn(); 
		ImGui::Text("#iter");
		ImGui::TableNextColumn(); 
		ImGui::Text("Curr Time [ms]");
		ImGui::TableNextColumn(); 
		ImGui::Text("Avg Time [ms]");
		ImGui::TableNextColumn(); 
		ImGui::Text("Total Time [m]");
		ImGui::TableNextColumn(); 
		ImGui::Text("lineSearch step size");
		ImGui::TableNextColumn(); 
		ImGui::Text("lineSearch #iter");
		//ImGui::TableAutoHeaders();
		ImGui::Separator();
		ImGui::PushItemWidth(80);
		for (auto&out : Outputs) {
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::PushID(id++);
			const int  i64_zero = 0, i64_max = 100000.0;
			ImGui::Text((modelName + std::to_string(out.ModelID)).c_str());
			ImGui::TableNextColumn();
			ImGui::Checkbox("##Max_Update", &out.minimizer->isUpdateLambdaWhenConverge);
			ImGui::TableNextColumn();
			ImGui::Checkbox("##On/Off", &out.minimizer->isAutoLambdaRunning);
			ImGui::TableNextColumn();
			ImGui::DragInt("##From", &(out.minimizer->autoLambda_from), 1, i64_zero, i64_max);
			ImGui::TableNextColumn();
			ImGui::DragInt("##count", &(out.minimizer->autoLambda_count), 1, i64_zero, i64_max, "2^%d");
			ImGui::TableNextColumn();
			ImGui::DragInt("##jump", &(out.minimizer->autoLambda_jump), 1, 1, i64_max);
			
			ImGui::TableNextColumn();
			ImGui::Text(std::to_string(out.minimizer->getNumiter()).c_str());
			ImGui::TableNextColumn();
			ImGui::Text(std::to_string(out.minimizer->timer_curr).c_str());
			ImGui::TableNextColumn();
			ImGui::Text(std::to_string(out.minimizer->timer_avg).c_str());
			ImGui::TableNextColumn();

			double tot_time = out.minimizer->timer_sum / 1000;
			int sec = int(tot_time) % 60;
			int min = (int(tot_time) - sec) / 60;

			ImGui::Text((std::to_string(min)+":" +std::to_string(sec)).c_str());
			ImGui::TableNextColumn();
			ImGui::Text(("2^" + std::to_string(int(log2(out.minimizer->init_step_size)))).c_str());
			ImGui::TableNextColumn();
			ImGui::Text(std::to_string(out.minimizer->linesearch_numiterations).c_str());
			ImGui::PopID();
		}
		ImGui::PopItemWidth();
		ImGui::EndTable();
	}

	ImGui::Text("");
	
	if (Outputs.size() != 0) {
		if (ImGui::BeginTable("Unconstrained weights table", Outputs[0].totalObjective->objectiveList.size() + 3, ImGuiTableFlags_Resizable))
		{
			ImGui::Separator();
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			ImGui::Text("ID");
			ImGui::TableNextColumn();
			ImGui::Text("Run");
			for (auto& obj : Outputs[0].totalObjective->objectiveList) {
				ImGui::TableNextColumn();
				ImGui::Text(obj->name.c_str());
			}
			ImGui::TableNextColumn();
			ImGui::Text("Remove Mesh");
			//ImGui::TableAutoHeaders();
			ImGui::Separator();
			
			
			for (int i = 0; i < Outputs.size(); i++) 
			{
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::Text((modelName + std::to_string(Outputs[i].ModelID)).c_str());
				ImGui::TableNextColumn();
				ImGui::PushID(id++);
				if (ImGui::Button("On/Off")) {
					if (Outputs[i].minimizer->is_running)
						stop_one_minimizer_thread(Outputs[i]);
					else
						start_one_minimizer_thread(Outputs[i]);
				}
				ImGui::PopID();
				ImGui::TableNextColumn();
				ImGui::PushItemWidth(80);
				for (auto& obj : Outputs[i].totalObjective->objectiveList) {
					ImGui::PushID(id++);
					ImGui::DragFloat("##w", &(obj->w), 0.05f, 0.0f, 100000.0f);
					auto SD = std::dynamic_pointer_cast<ObjectiveFunctions::Deformation::SymmetricDirichlet>(obj);
					auto fR = std::dynamic_pointer_cast<ObjectiveFunctions::Fabrication::RoundRadiuses>(obj);

					auto ABN = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxPlanar>(obj);
					auto AS = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxSphere>(obj);
					auto BN = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::Planar>(obj);

					if (obj->w) {
						if (fR != NULL) {
							ImGui::DragInt("min", &(fR->min));
							fR->min = fR->min < 1 ? 1 : fR->min;
							ImGui::DragInt("max", &(fR->max));
							fR->max = fR->max > fR->min ? fR->max : fR->min + 1;
							ImGui::DragFloat("alpha", &(fR->alpha), 0.001);
							Eigen::VectorXd Radiuses = Outputs[ActiveOutput].getRadiusOfSphere();
							if (ImGui::Button("update Alpha")) {
								fR->alpha = fR->max / Radiuses.maxCoeff();
							}
							ImGui::Text(("R max: " + std::to_string(Radiuses.maxCoeff() * fR->alpha)).c_str());
							ImGui::Text(("R min: " + std::to_string(Radiuses.minCoeff() * fR->alpha)).c_str());
							
						}

						if (ABN != NULL)
							ImGui::Combo("Function", (int*)(&(ABN->penaltyFunction)), "Quadratic\0Exponential\0Sigmoid\0\0");
						if (BN != NULL)
							ImGui::Combo("Function", (int*)(&(BN->penaltyFunction)), "Quadratic\0Exponential\0Sigmoid\0\0");
						if (AS != NULL)
							ImGui::Combo("Function", (int*)(&(AS->penaltyFunction)), "Quadratic\0Exponential\0Sigmoid\0\0");
						
						if (ABN != NULL && ABN->penaltyFunction == Cuda::PenaltyFunction::SIGMOID) {
							ImGui::Text(("2^" + std::to_string(int(log2(ABN->get_SigmoidParameter())))).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								ABN->Inc_SigmoidParameter();
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								ABN->Dec_SigmoidParameter();
							}
							const double  f64_zero = 0, f64_max = 100000.0;
							ImGui::DragScalar("w1", ImGuiDataType_Double, &(ABN->w1), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w2", ImGuiDataType_Double, &(ABN->w2), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w3", ImGuiDataType_Double, &(ABN->w3), 0.05f, &f64_zero, &f64_max);
						}
						if (AS != NULL && AS->penaltyFunction == Cuda::PenaltyFunction::SIGMOID) {
							ImGui::Text(("2^" + std::to_string(int(log2(AS->get_SigmoidParameter())))).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								AS->Inc_SigmoidParameter();
							}
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
							{
								AS->Dec_SigmoidParameter();
							}
							const double  f64_zero = 0, f64_max = 100000.0;
							ImGui::DragScalar("w1", ImGuiDataType_Double, &(AS->w1), 0.05f, &f64_zero, &f64_max);
							ImGui::DragScalar("w2", ImGuiDataType_Double, &(AS->w2), 0.05f, &f64_zero, &f64_max);
						}
						if (BN != NULL && BN->penaltyFunction == Cuda::PenaltyFunction::SIGMOID) {
							ImGui::Text(("2^" + std::to_string(int(log2(BN->get_SigmoidParameter())))).c_str());
							ImGui::SameLine();
							if (ImGui::Button("*", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
								BN->Inc_SigmoidParameter();
							ImGui::SameLine();
							if (ImGui::Button("/", ImVec2(ImGui::GetFrameHeight(), ImGui::GetFrameHeight())))
								BN->Dec_SigmoidParameter();
						}
					}
					ImGui::TableNextColumn();
					ImGui::PopID();
				}
				ImGui::PushID(id++);
				ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.0f, 0.0f, 1.0f));
				if (Outputs.size() > 1 && ImGui::Button("Remove"))
					remove_output(i);
				ImGui::PopStyleColor();
				ImGui::PopID();
				ImGui::PopItemWidth();
				//ImGui::TableNextRow();
			}	
			ImGui::EndTable();
		}
	}
	ImVec2 w_size = ImGui::GetWindowSize();
	energies_window_position = ImVec2(0.5 * global_screen_size[0] - 0.5 * w_size[0], global_screen_size[1] - w_size[1]);
	//close the window
	ImGui::End();
}

void MeshSimplificationPlugin::Draw_output_window()
{
	if (!outputs_window)
		return;
	for (auto& out : Outputs) 
	{
		ImGui::SetNextWindowSize(ImVec2(200, 300));
		ImGui::SetNextWindowPos(out.outputs_window_position);
		ImGui::Begin(("Output settings " + std::to_string(out.CoreID)).c_str(),
			NULL,
			ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove
		);
		ImGui::Checkbox("Update all models together", &isUpdateAll);
		CollapsingHeader_cores(viewer->core(out.CoreID), viewer->data(out.ModelID));
		CollapsingHeader_models(viewer->data(out.ModelID));

		ImGui::Text("Show:");
		if (ImGui::Checkbox("Norm", &(out.showFacesNorm)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showFacesNorm = out.showFacesNorm;
		ImGui::SameLine();
		if (ImGui::Checkbox("Norm Edges", &(out.showNormEdges)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showNormEdges = out.showNormEdges;
		if (ImGui::Checkbox("Sphere", &(out.showSphereCenters)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showSphereCenters = out.showSphereCenters;
		ImGui::SameLine();
		if (ImGui::Checkbox("Sphere Edges", &(out.showSphereEdges)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showSphereEdges = out.showSphereEdges;
		if (ImGui::Checkbox("Face Centers", &(out.showTriangleCenters)) && isUpdateAll)
			for (auto&oi : Outputs)
				oi.showTriangleCenters = out.showTriangleCenters;
		ImGui::End();
	}
}

void MeshSimplificationPlugin::Draw_results_window()
{
	if (!results_window)
		return;
	for (auto& out : Outputs)
	{
		bool bOpened2(true);
		ImColor c(text_color[0], text_color[1], text_color[2], 1.0f);
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0));
		ImGui::Begin(("Text " + std::to_string(out.CoreID)).c_str(), &bOpened2,
			ImGuiWindowFlags_NoTitleBar |
			ImGuiWindowFlags_NoResize |
			ImGuiWindowFlags_NoMove |
			ImGuiWindowFlags_NoScrollbar |
			ImGuiWindowFlags_NoScrollWithMouse |
			ImGuiWindowFlags_NoBackground |
			ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_NoSavedSettings |
			ImGuiWindowFlags_NoInputs |
			ImGuiWindowFlags_NoFocusOnAppearing |
			ImGuiWindowFlags_NoBringToFrontOnFocus);
		ImGui::SetWindowPos(out.results_window_position);
		ImGui::SetWindowSize(out.screen_size);
		ImGui::SetWindowCollapsed(false);

		ImGui::TextColored(c, (
			std::string("Num Faces: ") +
			std::to_string(InputModel().F.rows()) +
			std::string("\tNum Vertices: ") +
			std::to_string(InputModel().V.rows()) +
			std::string("\nGrad Size: ") +
			std::to_string(out.totalObjective->objectiveList[0]->grad.size) +
			std::string("\tNum Clusters: ") +
			std::to_string(out.clustering_faces_indices.size())
			).c_str());
		ImGui::TextColored(c, (std::string(out.totalObjective->name) + std::string(" energy ") + std::to_string(out.totalObjective->energy_value)).c_str());
		ImGui::TextColored(c, (std::string(out.totalObjective->name) + std::string(" gradient ") + std::to_string(out.totalObjective->gradient_norm)).c_str());
		for (auto& obj : out.totalObjective->objectiveList) {
			if (obj->w)
			{
				ImGui::TextColored(c, (std::string(obj->name) + std::string(" energy ") + std::to_string(obj->energy_value)).c_str());
				ImGui::TextColored(c, (std::string(obj->name) + std::string(" gradient ") + std::to_string(obj->gradient_norm)).c_str());
			}
		}
		ImGui::End();
		ImGui::PopStyleColor();
	}
}

void MeshSimplificationPlugin::clear_sellected_faces_and_vertices() 
{
	for (auto& o : Outputs) {
		o.Energy_auxSphere->Clear_HingesWeights();
		o.Energy_auxPlanar->Clear_HingesWeights();
		o.Energy_Planar->Clear_HingesWeights();
		o.Energy_PinChosenVertices->clearConstraints();
	}
}

void MeshSimplificationPlugin::update_parameters_for_all_cores() 
{
	if (!isUpdateAll)
		return;
	for (auto& core : viewer->core_list) 
	{
		int output_index = NOT_FOUND;
		for (int i = 0; i < Outputs.size(); i++)
			if (core.id == Outputs[i].CoreID)
				output_index = i;
		if (output_index == NOT_FOUND)
		{
			if (this->prev_camera_zoom != core.camera_zoom ||
				this->prev_camera_translation != core.camera_translation ||
				this->prev_trackball_angle.coeffs() != core.trackball_angle.coeffs()
				) 
			{
				for (auto& c : viewer->core_list) 
				{
					c.camera_zoom = core.camera_zoom;
					c.camera_translation = core.camera_translation;
					c.trackball_angle = core.trackball_angle;
				}	
				this->prev_camera_zoom = core.camera_zoom;
				this->prev_camera_translation = core.camera_translation;
				this->prev_trackball_angle = core.trackball_angle;
				for (auto&o : Outputs)
				{
					o.prev_camera_zoom = core.camera_zoom;
					o.prev_camera_translation = core.camera_translation;
					o.prev_trackball_angle = core.trackball_angle;
				}
			}
		}
		else 
		{
			if (Outputs[output_index].prev_camera_zoom != core.camera_zoom ||
				Outputs[output_index].prev_camera_translation != core.camera_translation ||
				Outputs[output_index].prev_trackball_angle.coeffs() != core.trackball_angle.coeffs()
				) 
			{
				for (auto& c : viewer->core_list) 
				{
					c.camera_zoom = core.camera_zoom;
					c.camera_translation = core.camera_translation;
					c.trackball_angle = core.trackball_angle;
				}	
				this->prev_camera_zoom = core.camera_zoom;
				this->prev_camera_translation = core.camera_translation;
				this->prev_trackball_angle = core.trackball_angle;
				for (auto&o : Outputs) 
				{
					o.prev_camera_zoom = core.camera_zoom;
					o.prev_camera_translation = core.camera_translation;
					o.prev_trackball_angle = core.trackball_angle;
				}	
			}
		}
	}
}

void MeshSimplificationPlugin::remove_output(const int output_index) 
{
	stop_all_minimizers_threads();
	viewer->erase_core(1 + output_index);
	viewer->erase_mesh(1 + output_index);
	Outputs.erase(Outputs.begin() + output_index);
	
	core_size = 1.0 / (Outputs.size() + 1.0);
	int frameBufferWidth, frameBufferHeight;
	glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
	post_resize(frameBufferWidth, frameBufferHeight);
}

void MeshSimplificationPlugin::add_output() 
{
	stop_all_minimizers_threads();
	const int index = Outputs.size();
	const int coreID = viewer->append_core(Eigen::Vector4f(0, 0, 0, 0) /*viewport*/);
	const int meshID = viewer->append_mesh();
	Outputs.push_back(GUIExtensions::MeshSimplificationData(coreID, meshID, viewer));
	core_size = 1.0 / (Outputs.size() + 1.0);
	
	viewer->data(Outputs[index].ModelID).clear();
	viewer->data(Outputs[index].ModelID).set_mesh(original_V, original_F);
	viewer->data(Outputs[index].ModelID).compute_normals();
	viewer->data(Outputs[index].ModelID).uniform_colors(
		Eigen::Vector3d(51.0 / 255.0, 43.0 / 255.0, 33.3 / 255.0),
		Eigen::Vector3d(255.0 / 255.0, 228.0 / 255.0, 58.0 / 255.0),
		Eigen::Vector3d(255.0 / 255.0, 235.0 / 255.0, 80.0 / 255.0));
	init_objective_functions(index);
	update_core_settings(original_V, original_F);
	int frameBufferWidth, frameBufferHeight;
	glfwGetFramebufferSize(viewer->window, &frameBufferWidth, &frameBufferHeight);
	post_resize(frameBufferWidth, frameBufferHeight);	

	assert(Outputs.size() == (viewer->data_list.size() - 1));
	assert(Outputs.size() == (viewer->core_list.size() - 1));
}

IGL_INLINE void MeshSimplificationPlugin::post_resize(int w, int h)
{
	if (!isPluginInitialized || !viewer)
		return;
	if (view == app_utils::View::HORIZONTAL) 
	{
		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, w - w * Outputs.size() * core_size, h);
		for (int i = 0; i < Outputs.size(); i++) 
		{
			Outputs[i].screen_position = ImVec2(w - w * (Outputs.size() - i) * core_size, 0);
			Outputs[i].screen_size = ImVec2(w * core_size, h);
			Outputs[i].results_window_position = Outputs[i].screen_position;
			Outputs[i].outputs_window_position = ImVec2(w - w * (Outputs.size() - (i + 1)) * core_size - 200, 0);
		}
	}
	if (view == app_utils::View::VERTICAL) 
	{
		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, Outputs.size() * h * core_size, w, h - Outputs.size() * h * core_size);
		for (int i = 0; i < Outputs.size(); i++) 
		{
			Outputs[i].screen_position = ImVec2(0, (Outputs.size() - i - 1) * h * core_size);
			Outputs[i].screen_size = ImVec2(w, h * core_size);
			Outputs[i].outputs_window_position = ImVec2(w-205, h - Outputs[i].screen_position[1] - Outputs[i].screen_size[1]);
			Outputs[i].results_window_position = ImVec2(0, Outputs[i].outputs_window_position[1]);
		}
	}
	if (view == app_utils::View::SHOW_INPUT_SCREEN_ONLY) 
	{
		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, w, h);
		for (auto&o : Outputs) 
		{
			o.screen_position = ImVec2(w, h);
			o.screen_size = ImVec2(0, 0);
			o.results_window_position = o.screen_position;
			//o.outputs_window_position = 
		}
	}
 	if (view >= app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0) 
	{
 		viewer->core(inputCoreID).viewport = Eigen::Vector4f(0, 0, 0, 0);
 		for (auto&o : Outputs) 
		{
 			o.screen_position = ImVec2(w, h);
 			o.screen_size = ImVec2(0, 0);
 			o.results_window_position = o.screen_position;
 		}
 		// what does this means?
 		Outputs[view - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].screen_position = ImVec2(0, 0);
 		Outputs[view - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].screen_size = ImVec2(w, h);
 		Outputs[view - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].results_window_position = ImVec2(w*0.8, 0);
 	}		
	for (auto& o : Outputs)
		viewer->core(o.CoreID).viewport = Eigen::Vector4f(o.screen_position[0], o.screen_position[1], o.screen_size[0] + 1, o.screen_size[1] + 1);
	energies_window_position = ImVec2(0.1 * w, 0.8 * h);
	global_screen_size = ImVec2(w, h);
}

IGL_INLINE bool MeshSimplificationPlugin::mouse_move(int mouse_x, int mouse_y)
{
	if (IsMouseDraggingAnyWindow)
		return true;	
	if (ui.isChoosingCluster()) {
		pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point);
		return true;
	}
	if (ui.isTranslatingVertex()) {
		Eigen::RowVector3d vertex_pos = OutputModel(ui.Output_Index).V.row(ui.Vertex_Index);
		Eigen::RowVector3d translation = app_utils::computeTranslation(mouse_x, ui.down_mouse_x, mouse_y, ui.down_mouse_y, vertex_pos, OutputCore(ui.Output_Index));
		for (auto& out : listOfOutputsToUpdate(ui.Output_Index))
			out.first.Energy_PinChosenVertices->translateConstraint(ui.Vertex_Index, translation);
		ui.down_mouse_x = mouse_x;
		ui.down_mouse_y = mouse_y;
		return true;
	}
	if (ui.isBrushingWeightInc() && pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
		double shift = (ui.ADD_DELETE == ADD) ? ADDING_WEIGHT_PER_HINGE_VALUE : -ADDING_WEIGHT_PER_HINGE_VALUE;
		const std::vector<int> brush_faces = Outputs[ui.Output_Index].FaceNeigh(ui.intersec_point.cast<double>(), brush_radius);
		for (auto& out : listOfOutputsToUpdate(ui.Output_Index)) {
			out.first.Energy_auxPlanar->Incr_HingesWeights(brush_faces, shift);
			out.first.Energy_Planar->Incr_HingesWeights(brush_faces, shift);
			out.first.Energy_auxSphere->Incr_HingesWeights(brush_faces, shift);
		}
		return true;
	}
	if (ui.isBrushingWeightDec() && pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
		const std::vector<int> brush_faces = Outputs[ui.Output_Index].FaceNeigh(ui.intersec_point.cast<double>(), brush_radius);
		for (auto& out : listOfOutputsToUpdate(ui.Output_Index)) {
			out.first.Energy_auxPlanar->setOne_HingesWeights(brush_faces);
			out.first.Energy_Planar->setOne_HingesWeights(brush_faces);
			out.first.Energy_auxSphere->setOne_HingesWeights(brush_faces);
		}
		return true;
	}
	
	int output_index, vertex_index;
	if (ui.isUsingDFS() && pick_vertex(output_index, vertex_index)) {
		ui.updateVerticesListOfDFS(InputModel().F, InputModel().V.rows(), vertex_index);
		return true;
	}

	if (ui.isUsingDFS() || ui.isBrushingWeightDec() || ui.isBrushingWeightInc())
		return true;
	return false;
}

std::vector<std::pair<GUIExtensions::MeshSimplificationData&, int>> MeshSimplificationPlugin::listOfOutputsToUpdate(const int out_index) {
	std::vector<std::pair<GUIExtensions::MeshSimplificationData&, int>> vec;
	if (out_index<0 || out_index>Outputs.size())
		return {};
	if (UserInterface_UpdateAllOutputs) {
		for (int i = 0; i < Outputs.size(); i++) {
			vec.push_back({ Outputs[i],i });
		}
		return vec;
	}
	return { { Outputs[out_index],out_index } };
}

IGL_INLINE bool MeshSimplificationPlugin::mouse_scroll(float delta_y) 
{
	if (IsMouseDraggingAnyWindow || ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
		return true;
	if (ui.isBrushing()) {
		brush_radius += delta_y * 0.005;
		brush_radius = std::max<float>(0.005, brush_radius);
		return true;
	}
	if (ui.isChoosingCluster()) {
		neighbor_distance += delta_y * 0.05;
		neighbor_distance = std::max<float>(0.005, neighbor_distance);
		return true;
	}
	return false;
}

IGL_INLINE bool MeshSimplificationPlugin::mouse_up(int button, int modifier) 
{
	IsMouseDraggingAnyWindow = false;

	int output_index, vertex_index;
	if (ui.isUsingDFS() && pick_vertex(output_index, vertex_index)) {
		ui.updateVerticesListOfDFS(InputModel().F, InputModel().V.rows(), vertex_index);
		for (auto& out : listOfOutputsToUpdate(output_index)) {
			out.first.Energy_auxPlanar->setZero_HingesWeights(ui.DFS_vertices_list);
			out.first.Energy_Planar->setZero_HingesWeights(ui.DFS_vertices_list);
			out.first.Energy_auxSphere->setZero_HingesWeights(ui.DFS_vertices_list);
		}
	}

	if (ui.isChoosingCluster() && pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
		std::vector<int> neigh_faces = Outputs[ui.Output_Index].getNeigh(neighbor_Type, InputModel().F, ui.Face_index, neighbor_distance);
		double shift = (ui.ADD_DELETE == ADD) ? 5 * ADDING_WEIGHT_PER_HINGE_VALUE : -5 * ADDING_WEIGHT_PER_HINGE_VALUE;
		for (auto& out : listOfOutputsToUpdate(ui.Output_Index)) {
			out.first.Energy_auxPlanar->Incr_HingesWeights(neigh_faces, shift);
			out.first.Energy_Planar->Incr_HingesWeights(neigh_faces, shift);
			out.first.Energy_auxSphere->Incr_HingesWeights(neigh_faces, shift);
		}
	}
	ui.clear();
	return false;
}

IGL_INLINE bool MeshSimplificationPlugin::mouse_down(int button, int modifier) 
{
	bool LeftClick = (button == GLFW_MOUSE_BUTTON_LEFT);
	bool RightClick = (button == GLFW_MOUSE_BUTTON_MIDDLE);
	if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
		IsMouseDraggingAnyWindow = true;
	ui.down_mouse_x = viewer->current_mouse_x;
	ui.down_mouse_y = viewer->current_mouse_y;
	
	if (ui.status == app_utils::UserInterfaceOptions::FIX_FACES && LeftClick) {
		//////..............
	}
	if (ui.status == app_utils::UserInterfaceOptions::FIX_FACES && RightClick) {
		//////..............
	}
	if (ui.status == app_utils::UserInterfaceOptions::FIX_VERTICES && LeftClick)
	{
		if (pick_vertex(ui.Output_Index, ui.Vertex_Index) && ui.Output_Index != INPUT_MODEL_SCREEN) {
			for (auto& out : listOfOutputsToUpdate(ui.Output_Index))
				out.first.Energy_PinChosenVertices->insertConstraint(ui.Vertex_Index, OutputModel(out.second).V);
			ui.isActive = true;
		}
	}
	if (ui.status == app_utils::UserInterfaceOptions::FIX_VERTICES && RightClick) {
		if (pick_vertex(ui.Output_Index, ui.Vertex_Index) && ui.Output_Index != INPUT_MODEL_SCREEN)
			for (auto& out : listOfOutputsToUpdate(ui.Output_Index))
				out.first.Energy_PinChosenVertices->eraseConstraint(ui.Vertex_Index);
		ui.clear();
	}
	if (ui.status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR && LeftClick) {
		if (pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
			ui.ADD_DELETE = ADD;
			ui.isActive = true;
		}
	}
	if (ui.status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR && RightClick) {
		if (pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
			ui.ADD_DELETE = DELETE;
			ui.isActive = true;
		}
	}
	if (ui.status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR && LeftClick) {
		if (pick_vertex(ui.Output_Index, ui.Vertex_Index)) {
			ui.DFS_Vertex_Index_FROM = ui.Vertex_Index;
			ui.ADD_DELETE = ADD;
			ui.isActive = true;
		}
	}
	if (ui.status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR && RightClick) {
		if (pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point)) {
			ui.ADD_DELETE = DELETE;
			ui.isActive = true;
		}
	}
	if (ui.status == app_utils::UserInterfaceOptions::ADJ_WEIGHTS && LeftClick) {
		ui.ADD_DELETE = ADD;
		ui.isActive = true;
		pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point);
	}
	if (ui.status == app_utils::UserInterfaceOptions::ADJ_WEIGHTS && RightClick) {
		ui.ADD_DELETE = DELETE;
		ui.isActive = true;
		pick_face(ui.Output_Index, ui.Face_index, ui.intersec_point);
	}

	return false;
}

IGL_INLINE bool MeshSimplificationPlugin::key_pressed(unsigned int key, int modifiers) 
{
	if ((key == 'c' || key == 'C') && modifiers == 1)
		clear_sellected_faces_and_vertices();
	if ((key == 'x' || key == 'X') && modifiers == 1) {
		if (face_coloring_Type == app_utils::Face_Colors::NO_COLORS) {
			face_coloring_Type = app_utils::Face_Colors::SPHERES_CLUSTERING;
			if (neighbor_Type == app_utils::Neighbor_Type::LOCAL_NORMALS)
				face_coloring_Type = app_utils::Face_Colors::NORMALS_CLUSTERING;
		}
		else if (face_coloring_Type == app_utils::Face_Colors::SIGMOID_PARAMETER) {
			face_coloring_Type = app_utils::Face_Colors::NO_COLORS;
		}	
		else {
			face_coloring_Type = app_utils::Face_Colors::SIGMOID_PARAMETER;
		}
	}
	if ((key == 'R' || key == 'r') && modifiers == 1) {
		for (auto& out : Outputs) {
			out.showNormEdges = out.showFacesNorm = false;
		}
		for (auto& d : viewer->data_list)
			d.show_faces = true;
	}
	if ((key == 'f' || key == 'F') && modifiers == 1) {
		for (auto& out : Outputs) {
			out.showNormEdges = out.showFacesNorm = true;
		}
		for (auto& d : viewer->data_list)
			d.show_faces = false;
	}
	if ((key == 'q' || key == 'Q') && modifiers == 1) 
	{
		neighbor_Type = app_utils::Neighbor_Type::LOCAL_NORMALS;
		face_coloring_Type = app_utils::Face_Colors::NORMALS_CLUSTERING;
		for (auto&out : Outputs) {
			out.showFacesNorm = false;
			out.showSphereEdges = out.showNormEdges = 
				out.showTriangleCenters = out.showSphereCenters = false;
		}
		for (GUIExtensions::MeshSimplificationData& out : Outputs) {
			auto AS = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxSphere>(out.totalObjective->objectiveList[0]);
			auto AP = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxPlanar>(out.totalObjective->objectiveList[1] );
			auto BN = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::Planar>(out.totalObjective->objectiveList[2]);
			AP->w = 0;
			BN->w = 1.6;
			AS->w = 0;
		}
	}
	if ((key == 'e' || key == 'E') && modifiers == 1)
	{
		neighbor_Type = app_utils::Neighbor_Type::LOCAL_NORMALS;
		face_coloring_Type = app_utils::Face_Colors::NORMALS_CLUSTERING;
		for (auto& out : Outputs) {
			out.showFacesNorm = true;
			out.showSphereEdges = out.showNormEdges =
				out.showTriangleCenters = out.showSphereCenters = false;
		}
		for (GUIExtensions::MeshSimplificationData& out : Outputs) {
			auto AS = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxSphere>(out.totalObjective->objectiveList[0]);
			auto AP = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxPlanar>(out.totalObjective->objectiveList[1]);
			auto BN = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::Planar>(out.totalObjective->objectiveList[2]);
			AP->w = 1.6;
			BN->w = 0;
			AS->w = 0;
		}
	}
	if ((key == 'w' || key == 'W') && modifiers == 1) 
	{
		neighbor_Type = app_utils::Neighbor_Type::LOCAL_SPHERE;
		face_coloring_Type = app_utils::Face_Colors::SPHERES_CLUSTERING;
		initSphereAuxVariables = OptimizationUtils::InitSphereAuxVariables::MINUS_NORMALS;
		init_aux_variables();
		for (auto&out : Outputs) {
			out.showSphereCenters = true;
			out.showSphereEdges = out.showNormEdges =
				out.showTriangleCenters = out.showFacesNorm = false;
		}
		for (GUIExtensions::MeshSimplificationData& out : Outputs)
		{
			for (auto& obj : out.totalObjective->objectiveList) 
			{
				auto AS = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxSphere>(out.totalObjective->objectiveList[0]);
				auto AP = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::AuxPlanar>(out.totalObjective->objectiveList[1]);
				auto BN = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::Planar>(out.totalObjective->objectiveList[2]);
				AP->w = 0;
				BN->w = 0;
				AS->w = 1.6;
			}
		}
	}
	
	if ((key == ' ') && modifiers == 1)
		isMinimizerRunning ? stop_all_minimizers_threads() : start_all_minimizers_threads();
	
	return ImGuiPlugin::key_pressed(key, modifiers);
}

IGL_INLINE bool MeshSimplificationPlugin::key_down(int key, int modifiers)
{
	if (key == '1')
		ui.status = app_utils::UserInterfaceOptions::FIX_VERTICES;
	else if (key == '2')
		ui.status = app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR;
	else if (key == '3')
		ui.status = app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR;
	else if (key == '4')
		ui.status = app_utils::UserInterfaceOptions::ADJ_WEIGHTS;
	else if (key == '5')
		ui.status = app_utils::UserInterfaceOptions::FIX_FACES;
	
	return ImGuiPlugin::key_down(key, modifiers);
}

IGL_INLINE bool MeshSimplificationPlugin::key_up(int key, int modifiers)
{
	ui.status = app_utils::UserInterfaceOptions::NONE;
	return ImGuiPlugin::key_up(key, modifiers);
}

IGL_INLINE void MeshSimplificationPlugin::shutdown()
{
	stop_all_minimizers_threads();
	ImGuiPlugin::shutdown();
}

void MeshSimplificationPlugin::draw_brush_sphere() 
{
	if (!ui.isBrushing())
		return;
	//prepare brush sphere
	const int samples = 100;
	Eigen::MatrixXd sphere(samples * samples, 3);
	Eigen::RowVector3d center = ui.intersec_point.cast<double>().transpose();
	int i, j;
	for (double alfa = 0, i = 0; alfa < 360; i++, alfa += int(360 / samples)) {
		for (double beta = 0, j = 0; beta < 360; j++, beta += int(360 / samples)) {
			Eigen::RowVector3d dir;
			dir << sin(alfa), cos(alfa)* cos(beta), sin(beta)* cos(alfa);
			if (i + samples * j < sphere.rows())
				sphere.row(i + samples * j) = dir * brush_radius + center;
		}
	}
	//update data for cores
	OutputModel(ui.Output_Index).add_points(sphere, ui.getBrushColor(model_color));
}

IGL_INLINE bool MeshSimplificationPlugin::pre_draw() 
{
	for (auto& out : Outputs)
		update_data_from_minimizer();
	update_parameters_for_all_cores();

	//Update Faces Colors
	follow_and_mark_selected_faces();
	InputModel().set_colors(Outputs[ActiveOutput].color_per_face);
	for (int i = 0; i < Outputs.size(); i++)
		OutputModel(i).set_colors(Outputs[i].color_per_face);

	//Update Vertices Colors
	for (int oi = 0; oi < Outputs.size(); oi++) {
		auto& m = OutputModel(oi);
		auto& o = Outputs[oi];
		auto& AS = Outputs[oi].Energy_auxSphere;
		m.point_size = 10;
		m.clear_points();
		m.clear_edges();

		if (ui.isTranslatingVertex())
			m.add_points(m.V.row(ui.Vertex_Index), Dragged_vertex_color.cast<double>().transpose());
		for (auto vi : o.Energy_PinChosenVertices->getConstraintsIndices())
			m.add_points(m.V.row(vi), Fixed_vertex_color.cast<double>().transpose());
		if (o.showFacesNorm)
			m.add_points(o.getFacesNorm(), o.color_per_face_norm);
		if (o.showTriangleCenters)
			m.add_points(o.getCenterOfFaces(), o.color_per_vertex_center);
		if (o.showSphereCenters)
			m.add_points(o.getCenterOfSphere(), o.color_per_sphere_center);
		if (o.showSphereEdges)
			m.add_edges(o.getCenterOfFaces(), o.getSphereEdges(), o.color_per_sphere_edge);
		if (o.showNormEdges)
			m.add_edges(o.getCenterOfFaces(), o.getFacesNorm(), o.color_per_norm_edge);
			
		// Update Vertices colors for UI sigmoid weights
		int num_hinges = AS->mesh_indices.num_hinges;
		const Eigen::VectorXi& x0_index = AS->x0_GlobInd;
		const Eigen::VectorXi& x1_index = AS->x1_GlobInd;
		double* hinge_val = AS->weight_PerHinge.host_arr;
		std::set<int> points_indices;
		for (int hi = 0; hi < num_hinges; hi++) {
			if (hinge_val[hi] == 0) {
				points_indices.insert(x0_index[hi]);
				points_indices.insert(x1_index[hi]);
			}
		}
		Eigen::MatrixXd points_pos(points_indices.size(), 3);
		auto& iter = points_indices.begin();
		for (int i = 0; i < points_pos.rows(); i++) {
			int v_index = *(iter++);
			points_pos.row(i) = m.V.row(v_index);
		}
		auto color = ui.colorM.cast<double>().replicate(1, points_indices.size()).transpose();
		m.add_points(points_pos, color);
	}

	for (auto& out : listOfOutputsToUpdate(ui.Output_Index)) {
		if (ui.DFS_vertices_list.size()) {
			Eigen::MatrixXd points_pos(ui.DFS_vertices_list.size(), 3);
			int i = 0;
			for (int v_index : ui.DFS_vertices_list)
				points_pos.row(i++) = OutputModel(out.second).V.row(v_index);
			OutputModel(out.second).add_points(points_pos, ui.colorTry.cast<double>().transpose());
		}
	}

	draw_brush_sphere();
	InputModel().point_size = OutputModel(ActiveOutput).point_size;
	InputModel().set_points(OutputModel(ActiveOutput).points.leftCols(3), OutputModel(ActiveOutput).points.rightCols(3));
	return ImGuiPlugin::pre_draw();
}

void MeshSimplificationPlugin::change_minimizer_type(Cuda::OptimizerType type)
{
	optimizer_type = type;
	stop_all_minimizers_threads();
	init_aux_variables();
	for (int i = 0; i < Outputs.size(); i++)
		Outputs[i].minimizer->Optimizer_type = optimizer_type;
}

void MeshSimplificationPlugin::update_core_settings(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
	for (int i = 0; i < viewer->core_list.size(); i++) {
		viewer->core_list[i].align_camera_center(V, F);
		viewer->core_list[i].trackball_angle = Eigen::Quaternionf::Identity();
		viewer->core_list[i].orthographic = false;
		viewer->core_list[i].set_rotation_type(igl::opengl::ViewerCore::RotationType(1));
		viewer->core_list[i].is_animating = true;
		viewer->core_list[i].background_color = Eigen::Vector4f(1, 1, 1, 0);
		viewer->core_list[i].lighting_factor = 0.5;
	}
	for (auto& data : viewer->data_list)
		for (auto& out : Outputs)
			data.copy_options(viewer->core(inputCoreID), viewer->core(out.CoreID));
	for (auto& core : viewer->core_list)
		for (auto& data : viewer->data_list)
			viewer->data(data.id).set_visible(false, core.id);
	InputModel().set_visible(true, inputCoreID);
	for (int i = 0; i < Outputs.size(); i++)
		OutputModel(i).set_visible(true, Outputs[i].CoreID);
}

void MeshSimplificationPlugin::follow_and_mark_selected_faces() 
{
	for (int i = 0; i < Outputs.size(); i++) {
		Outputs[i].initFaceColors(
			InputModel().F.rows(),
			center_sphere_color,
			center_vertex_color,
			Color_sphere_edges,
			Color_normal_edge,
			face_norm_color);

		UpdateEnergyColors(i);
		//Mark the selected faces by brush
		auto& AS = Outputs[i].Energy_auxSphere;
		for (int hi = 0; hi < AS->mesh_indices.num_hinges; hi++) {
			const int f0 = AS->hinges_faceIndex[hi][0];
			const int f1 = AS->hinges_faceIndex[hi][1];
			if (AS->weight_PerHinge.host_arr[hi] > 1) {
				const double alpha = (AS->weight_PerHinge.host_arr[hi] - 1.0f) / MAX_WEIGHT_PER_HINGE_VALUE;
				Outputs[i].shiftFaceColors(f0, alpha, model_color, ui.colorP);
				Outputs[i].shiftFaceColors(f1, alpha, model_color, ui.colorP);
			}
		}
	}

	//Mark the highlighted face & neighbors
	if (ui.isChoosingCluster()) {
		std::vector<int> neigh = Outputs[ui.Output_Index].getNeigh(neighbor_Type, InputModel().F, ui.Face_index, neighbor_distance);
		for (int fi : neigh)
			Outputs[ui.Output_Index].setFaceColors(fi, Neighbors_Highlighted_face_color.cast<double>());
		Outputs[ui.Output_Index].setFaceColors(ui.Face_index, Highlighted_face_color.cast<double>());
	}
		
	for (auto& o:Outputs) {
		//Mark the clusters if needed
		if (clusteringType == app_utils::ClusteringType::NO_CLUS && (face_coloring_Type == app_utils::Face_Colors::NORMALS_CLUSTERING || face_coloring_Type == app_utils::Face_Colors::SPHERES_CLUSTERING)) {
			Eigen::MatrixX3d P = o.getFacesNormals();
			if (face_coloring_Type == app_utils::Face_Colors::SPHERES_CLUSTERING) {
				Eigen::MatrixXd C = o.getCenterOfSphere();
				Eigen::VectorXd R = o.getRadiusOfSphere();
				for (int fi = 0; fi < C.rows(); fi++)
					P.row(fi) << C(fi, 0) * R(fi), C(fi, 1), C(fi, 2);
			}
			Eigen::RowVector3d Pmin(P.col(0).minCoeff(), P.col(1).minCoeff(), P.col(2).minCoeff());
			Eigen::RowVector3d Pmax(P.col(0).maxCoeff(), P.col(1).maxCoeff(), P.col(2).maxCoeff());
			for (int fi = 0; fi < P.rows(); fi++) {
				//set the values in the range [0, 1]
				for (int xyz = 0; xyz < 3; xyz++) {
					double range = Pmax(xyz) - Pmin(xyz);
					if (range < 0.01)
						P(fi, xyz) = 0.5;
					else
						P(fi, xyz) = (P(fi, xyz) - Pmin(xyz)) / range;
				}	
				//Add Brightness according to user weight...
				for (int col = 0; col < 3; col++)
					P(fi, col) = (clustering_brightness_w * P(fi, col)) + (1 - clustering_brightness_w);
				//set faces colors
				o.setFaceColors(fi, P.row(fi));
			}
			o.clustering_faces_colors = P;
			o.clustering_faces_indices = {};
		}
		else if (clusteringType != app_utils::ClusteringType::NO_CLUS && o.clustering_faces_indices.size() &&
			(face_coloring_Type == app_utils::Face_Colors::NORMALS_CLUSTERING || face_coloring_Type == app_utils::Face_Colors::SPHERES_CLUSTERING)) 
		{
			o.clustering_colors.getFacesColors(o.clustering_faces_indices, InputModel().F.rows(), clustering_brightness_w, o.clustering_faces_colors);
			//set faces colors
			for (int fi = 0; fi < InputModel().F.rows(); fi++)
				o.setFaceColors(fi, o.clustering_faces_colors.row(fi).transpose());
		}
		else if (face_coloring_Type == app_utils::Face_Colors::SIGMOID_PARAMETER) {
			auto& AS = o.Energy_auxSphere;
			for (int hi = 0; hi < AS->mesh_indices.num_hinges; hi++) {
				const int f0 = AS->hinges_faceIndex[hi][0];
				const int f1 = AS->hinges_faceIndex[hi][1]; 
				const double log_minus_w = -log2(AS->Sigmoid_PerHinge.host_arr[hi]);
				const double alpha = log_minus_w / MAX_SIGMOID_PER_HINGE_VALUE;
				o.shiftFaceColors(f0, alpha, model_color, ui.colorP);
				o.shiftFaceColors(f1, alpha, model_color, ui.colorP);
			}
		}

		if (isChecking_SelfIntersection) {
			for (int fi = 0; fi < InputModel().F.rows(); fi++)
				o.setFaceColors(fi, model_color.cast<double>());
			for (auto& p : o.SelfIntersection_pairs) {
				o.setFaceColors(p.first, Eigen::Vector3d(1, 0, 0));
				o.setFaceColors(p.second, Eigen::Vector3d(0, 1, 0));
			}
		}
		
		if (isChecking_FlippedFaces) {
			for (int fi = 0; fi < InputModel().F.rows(); fi++)
				o.setFaceColors(fi, model_color.cast<double>());
			for (auto& p : o.flippedFaces_pairs) {
				o.setFaceColors(p.first, Eigen::Vector3d(1, 0, 0));
				o.setFaceColors(p.second, Eigen::Vector3d(0, 1, 0));
			}
		}
	}
}
	
igl::opengl::ViewerData& MeshSimplificationPlugin::InputModel() 
{
	return viewer->data(inputModelID);
}

igl::opengl::ViewerData& MeshSimplificationPlugin::OutputModel(const int index) 
{
	return viewer->data(Outputs[index].ModelID);
}

igl::opengl::ViewerCore& MeshSimplificationPlugin::InputCore()
{
	return viewer->core(inputCoreID);
}

igl::opengl::ViewerCore& MeshSimplificationPlugin::OutputCore(const int index) 
{
	return viewer->core(Outputs[index].CoreID);
}

bool MeshSimplificationPlugin::pick_face(int& out_ind, int& f_ind, Eigen::Vector3f& intersec_point)
{
	f_ind = pick_face_per_core(InputModel().V, InputModel().F, app_utils::View::SHOW_INPUT_SCREEN_ONLY, intersec_point);
	out_ind = INPUT_MODEL_SCREEN;
	for (int i = 0; i < Outputs.size(); i++)
	{
		if (f_ind == NOT_FOUND)
		{
			f_ind = pick_face_per_core(OutputModel(i).V, OutputModel(i).F, app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0 + i, intersec_point);
			out_ind = i;
		}
	}
	return (f_ind != NOT_FOUND);
}

int MeshSimplificationPlugin::pick_face_per_core(
	Eigen::MatrixXd& V, 
	Eigen::MatrixXi& F, 
	int CoreIndex, 
	Eigen::Vector3f& intersec_point) 
{
	// Cast a ray in the view direction starting from the mouse position
	int CoreID;
	if (CoreIndex == app_utils::View::SHOW_INPUT_SCREEN_ONLY)
		CoreID = inputCoreID;
	else
		CoreID = Outputs[CoreIndex - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].CoreID;
	double x = viewer->current_mouse_x;
	double y = viewer->core(CoreID).viewport(3) - viewer->current_mouse_y;
	if (view == app_utils::View::VERTICAL) 
	{
		y = (viewer->core(inputCoreID).viewport(3) / core_size) - viewer->current_mouse_y;
	}
	Eigen::RowVector3d pt;
	Eigen::Matrix4f modelview = viewer->core(CoreID).view;
	int vi = NOT_FOUND;
	std::vector<igl::Hit> hits;
	igl::unproject_in_mesh(Eigen::Vector2f(x, y), viewer->core(CoreID).view,
		viewer->core(CoreID).proj, viewer->core(CoreID).viewport, V, F, pt, hits);
	Eigen::Vector3f s, dir;
	igl::unproject_ray(Eigen::Vector2f(x, y), viewer->core(CoreID).view,
		viewer->core(CoreID).proj, viewer->core(CoreID).viewport, s, dir);
	int fi = NOT_FOUND;
	if (hits.size() > 0) 
	{
		fi = hits[0].id;
		intersec_point = s + dir * hits[0].t;
	}
	return fi;
}

bool MeshSimplificationPlugin::pick_vertex(int& o_ind, int& v_index) {
	v_index = pick_vertex_per_core(InputModel().V, InputModel().F, app_utils::View::SHOW_INPUT_SCREEN_ONLY);
	o_ind = INPUT_MODEL_SCREEN;
	for (int i = 0; i < Outputs.size(); i++) {
		if (v_index == NOT_FOUND) {
			v_index = pick_vertex_per_core(OutputModel(i).V, OutputModel(i).F, app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0 + i);
			o_ind = i;
		}
	}
	return (v_index != NOT_FOUND);
}

int MeshSimplificationPlugin::pick_vertex_per_core(
	Eigen::MatrixXd& V, 
	Eigen::MatrixXi& F, 
	int CoreIndex) 
{
	// Cast a ray in the view direction starting from the mouse position
	int CoreID;
	if (CoreIndex == app_utils::View::SHOW_INPUT_SCREEN_ONLY)
		CoreID = inputCoreID;
	else
		CoreID = Outputs[CoreIndex - app_utils::View::SHOW_OUTPUT_SCREEN_ONLY_0].CoreID;
	double x = viewer->current_mouse_x;
	double y = viewer->core(CoreID).viewport(3) - viewer->current_mouse_y;
	if (view == app_utils::View::VERTICAL) {
		y = (viewer->core(inputCoreID).viewport(3) / core_size) - viewer->current_mouse_y;
	}
	Eigen::Matrix<double, 3, 1, 0, 3, 1> pt;
	Eigen::Matrix4f modelview = viewer->core(CoreID).view;
	int vi = NOT_FOUND;
	std::vector<igl::Hit> hits;
	unproject_in_mesh(
		Eigen::Vector2f(x, y), 
		viewer->core(CoreID).view,
		viewer->core(CoreID).proj, 
		viewer->core(CoreID).viewport, 
		V, 
		F, 
		pt, 
		hits
	);
	if (hits.size() > 0) 
	{
		int fi = hits[0].id;
		Eigen::RowVector3d bc;
		bc << 1.0 - hits[0].u - hits[0].v, hits[0].u, hits[0].v;
		bc.maxCoeff(&vi);
		vi = F(fi, vi);
	}
	return vi;
}

void MeshSimplificationPlugin::checkGradients()
{
	stop_all_minimizers_threads();
	for (auto& o: Outputs) 
	{
		Eigen::VectorXd testX = Eigen::VectorXd::Random(o.totalObjective->objectiveList[0]->grad.size);
		o.totalObjective->checkGradient(testX);
		for (auto const &objective : o.totalObjective->objectiveList)
			objective->checkGradient(testX);
	}
}

void MeshSimplificationPlugin::update_data_from_minimizer()
{	
	for (int i = 0; i < Outputs.size(); i++)
	{
		Eigen::MatrixXd V(original_V.rows(), 3);
		auto& o = Outputs[i];
		o.minimizer->get_data(V, o.center_of_sphere, o.radiuses, o.normals);
		o.center_of_faces = OptimizationUtils::center_per_triangle(V, InputModel().F);

		Eigen::MatrixX3d N;
		igl::per_face_normals(V, OutputModel(i).F, N);
		auto BN = std::dynamic_pointer_cast<ObjectiveFunctions::Panels::Planar>(Outputs[i].totalObjective->objectiveList[2]);
		if (BN->w != 0) {
			o.normals = N;
		}
		
		OutputModel(i).set_vertices(V);
		OutputModel(i).compute_normals();
	}
}

void MeshSimplificationPlugin::stop_all_minimizers_threads() {
	for (auto& o : Outputs)
		stop_one_minimizer_thread(o);
}

void MeshSimplificationPlugin::stop_one_minimizer_thread(const GUIExtensions::MeshSimplificationData o) {
	if (o.minimizer->is_running)
		o.minimizer->stop();
	while (o.minimizer->is_running);

	isMinimizerRunning = is_Any_Minizer_running();
}
void MeshSimplificationPlugin::start_all_minimizers_threads() {
	for (auto& o : Outputs)
		start_one_minimizer_thread(o);
}

void MeshSimplificationPlugin::start_one_minimizer_thread(const GUIExtensions::MeshSimplificationData o) {
	stop_one_minimizer_thread(o);
	std::thread minimizer_thread1 = std::thread(&NumericalOptimizations::Basic::run_new, o.minimizer.get());
	std::thread minimizer_thread2 = std::thread(&NumericalOptimizations::Basic::RunSymmetricDirichletGradient, o.minimizer.get());
	minimizer_thread1.detach();
	minimizer_thread2.detach();
	
	isMinimizerRunning = true;
}
bool MeshSimplificationPlugin::is_Any_Minizer_running() {
	for (auto&o : Outputs)
		if (o.minimizer->is_running)
			return true;
	return false;
}

void MeshSimplificationPlugin::init_aux_variables() 
{
	stop_all_minimizers_threads();
	if (InitMinimizer_NeighLevel_From < 1)
		InitMinimizer_NeighLevel_From = 1;
	if (InitMinimizer_NeighLevel_From > InitMinimizer_NeighLevel_To)
		InitMinimizer_NeighLevel_To = InitMinimizer_NeighLevel_From;
	for (int i = 0; i < Outputs.size(); i++)
		Outputs[i].initMinimizers(
			OutputModel(i).V,
			OutputModel(i).F,
			initSphereAuxVariables,
			InitMinimizer_NeighLevel_From,
			InitMinimizer_NeighLevel_To,
			radius_length_minus_normal);
}

void MeshSimplificationPlugin::run_one_minimizer_iter() 
{
	stop_all_minimizers_threads();
	for (auto& o : Outputs)
		o.minimizer->run_one_iteration();
}

void MeshSimplificationPlugin::init_objective_functions(const int index)
{
	Eigen::MatrixXd V = OutputModel(index).V;
	Eigen::MatrixX3i F = OutputModel(index).F;
	if (V.rows() == 0 || F.rows() == 0)
		return;

	Outputs[index].minimizer = std::make_shared<NumericalOptimizations::Basic>(Outputs[index].CoreID);
	Outputs[index].minimizer->lineSearch_type = linesearch_type;
	Outputs[index].minimizer->Optimizer_type = optimizer_type;

	// initialize the energy
	std::cout << Utils::ConsoleColor::yellow << "-------Energies, begin-------" << std::endl;
	std::shared_ptr <ObjectiveFunctions::Panels::AuxPlanar> auxPlanar = std::make_unique<ObjectiveFunctions::Panels::AuxPlanar>(V, F, Cuda::PenaltyFunction::SIGMOID);
	Outputs[index].Energy_auxPlanar = auxPlanar;
	std::shared_ptr <ObjectiveFunctions::Panels::AuxSphere> auxSphere = std::make_unique<ObjectiveFunctions::Panels::AuxSphere>(V, F, Cuda::PenaltyFunction::SIGMOID);
	Outputs[index].Energy_auxSphere = auxSphere;
	std::shared_ptr <ObjectiveFunctions::Deformation::STVK> stvk = std::make_unique<ObjectiveFunctions::Deformation::STVK>(V, F);
	std::shared_ptr <ObjectiveFunctions::Deformation::SymmetricDirichlet> sdenergy = std::make_unique<ObjectiveFunctions::Deformation::SymmetricDirichlet>(V, F);
	std::shared_ptr <ObjectiveFunctions::Deformation::PinVertices> fixAllVertices = std::make_unique<ObjectiveFunctions::Deformation::PinVertices>(V, F);
	std::shared_ptr <ObjectiveFunctions::Fabrication::RoundRadiuses> FixRadius = std::make_unique<ObjectiveFunctions::Fabrication::RoundRadiuses>(V, F);
	std::shared_ptr <ObjectiveFunctions::Deformation::UniformSmoothness> uniformSmoothness = std::make_unique<ObjectiveFunctions::Deformation::UniformSmoothness>(V, F);
	std::shared_ptr <ObjectiveFunctions::Panels::Planar> planar = std::make_unique<ObjectiveFunctions::Panels::Planar>(V, F, Cuda::PenaltyFunction::SIGMOID);
	Outputs[index].Energy_Planar = planar;

	//Add User Interface Energies
	auto pinChosenVertices = std::make_shared<ObjectiveFunctions::Deformation::PinChosenVertices>(V, F);
	Outputs[index].Energy_PinChosenVertices = pinChosenVertices;

	//init total objective
	Outputs[index].totalObjective = std::make_shared<ObjectiveFunctions::Total>(V, F);
	Outputs[index].totalObjective->objectiveList.clear();
	auto add_obj = [&](std::shared_ptr< ObjectiveFunctions::Basic> obj)
	{
		Outputs[index].totalObjective->objectiveList.push_back(move(obj));
	};
	add_obj(auxSphere);
	add_obj(auxPlanar);
	add_obj(planar);
	add_obj(stvk);
	add_obj(sdenergy);
	add_obj(fixAllVertices);
	add_obj(pinChosenVertices);
	add_obj(FixRadius);
	add_obj(uniformSmoothness);
	std::cout  << "-------Energies, end-------" << Utils::ConsoleColor::white << std::endl;
	init_aux_variables();
}

void MeshSimplificationPlugin::UpdateEnergyColors(const int index) 
{
	int numF = OutputModel(index).F.rows();
	Eigen::VectorXd DistortionPerFace(numF);
	DistortionPerFace.setZero();
	if (faceColoring_type == 0) { // No colors
		DistortionPerFace.setZero();
	}
	else if (faceColoring_type == 1) { // total energy
		for (auto& obj: Outputs[index].totalObjective->objectiveList) {
			// calculate the distortion over all the energies
			if ((obj->Efi.size() != 0) && (obj->w != 0))
				DistortionPerFace += obj->Efi * obj->w;
		}
	}
	else {
		auto& obj = Outputs[index].totalObjective->objectiveList[faceColoring_type - 2];
		if ((obj->Efi.size() != 0) && (obj->w != 0))
			DistortionPerFace = obj->Efi * obj->w;
	}
	Eigen::VectorXd alpha_vec = DistortionPerFace / (Max_Distortion+1e-8);
	Eigen::VectorXd beta_vec = Eigen::VectorXd::Ones(numF) - alpha_vec;
	Eigen::MatrixXd alpha(numF, 3), beta(numF, 3);
	alpha = alpha_vec.replicate(1, 3);
	beta = beta_vec.replicate(1, 3);
	//calculate low distortion color matrix
	Eigen::MatrixXd LowDistCol = model_color.cast <double>().replicate(1, numF).transpose();
	//calculate high distortion color matrix
	Eigen::MatrixXd HighDistCol = Vertex_Energy_color.cast <double>().replicate(1, numF).transpose();
	Outputs[index].color_per_face = beta.cwiseProduct(LowDistCol) + alpha.cwiseProduct(HighDistCol);
}
