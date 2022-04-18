#pragma once

#ifndef RDS_PLUGINS_BASIC_MENU_H
#define RDS_PLUGINS_BASIC_MENU_H

#include "optimization_output.h"
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>

class deformation_plugin : public igl::opengl::glfw::imgui::ImGuiPlugin
{
public:
	Eigen::MatrixXd original_V;
	Eigen::MatrixXi original_F;
	bool isChecking_SelfIntersection = false;
	float flippedFaces_epsilon = 0.01;
	float SelfIntersection_epsilon = 0.001;
	bool isChecking_FlippedFaces = false;
	int ActiveOutput = 0;
	int InitMinimizer_NeighLevel_From = 1;
	int InitMinimizer_NeighLevel_To = 10;
	bool CollapsingHeader_curr[9], CollapsingHeader_prev[9], CollapsingHeader_change;
	bool outputs_window, results_window, energies_window;
	OptimizationUtils::InitSphereAuxVariables initSphereAuxVariables;
	float Max_Distortion;
	float neighbor_distance, brush_radius;
	bool isUpdateAll;
	bool isMinimizerRunning, IsMouseDraggingAnyWindow;
	int faceColoring_type;
	float Clustering_MSE_Threshold = 0.001;
	app_utils::ClusteringType clusteringType = app_utils::ClusteringType::NO_CLUS;
	Cuda::OptimizerType optimizer_type;
	OptimizationUtils::LineSearch linesearch_type;
	float constantStep_LineSearch;
	Eigen::Vector3f
		Highlighted_face_color,
		center_sphere_color,
		center_vertex_color,
		face_norm_color,
		Color_sphere_edges,
		Color_normal_edge,
		Neighbors_Highlighted_face_color,
		Fixed_vertex_color,
		model_color,
		Dragged_vertex_color,
		Vertex_Energy_color,
		text_color;
	float core_size;
	app_utils::Face_Colors face_coloring_Type;
	float clustering_brightness_w;
	app_utils::Neighbor_Type neighbor_Type;
	std::vector<OptimizationOutput> Outputs;
	float prev_camera_zoom;
	Eigen::Vector3f prev_camera_translation;
	Eigen::Quaternionf prev_trackball_angle;
	std::string modelName;
	int inputCoreID, inputModelID;
	app_utils::View view;

	//UI variables
	UI ui;

	ImVec2 energies_window_position, global_screen_size;
	bool UserInterface_UpdateAllOutputs;
	float radius_length_minus_normal = 0.1;
public:
	deformation_plugin();
	~deformation_plugin(){}

	// callbacks
	void load_first_mesh(const std::string& name, const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
	IGL_INLINE virtual void init(igl::opengl::glfw::Viewer *_viewer) override;
	IGL_INLINE virtual void post_resize(int w, int h) override;
	IGL_INLINE virtual bool mouse_move(int mouse_x, int mouse_y) override;
	IGL_INLINE virtual bool mouse_down(int button, int modifier) override;
	IGL_INLINE virtual bool mouse_up(int button, int modifier) override;
	IGL_INLINE virtual bool mouse_scroll(float delta_y) override;
	IGL_INLINE virtual bool pre_draw() override;
	IGL_INLINE virtual void shutdown() override;
	IGL_INLINE virtual bool key_pressed(unsigned int key, int modifiers) override;
	IGL_INLINE virtual bool key_down(int key, int modifiers) override;
	IGL_INLINE virtual bool key_up(int key, int modifiers) override;
			
	//Draw Collapsing Headers
	void CollapsingHeader_cores(igl::opengl::ViewerCore& core, igl::opengl::ViewerData& data);
	void CollapsingHeader_models(igl::opengl::ViewerData& data);
	void CollapsingHeader_minimizer();
	void CollapsingHeader_screen();
	void CollapsingHeader_face_coloring();
	void CollapsingHeader_clustering();
	void CollapsingHeader_fabrication();
	void CollapsingHeader_measures();
	void CollapsingHeader_user_interface();
	void CollapsingHeader_colors();
	void CollapsingHeader_update();
	
	//Draw window
	void Draw_results_window();
	void Draw_energies_window();
	void Draw_output_window();
	
	//Pick faces & vertices and highlight them
	bool pick_face(int& output_index, int& face_index, Eigen::Vector3f& intersec_point);
	int pick_face_per_core(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int LR, Eigen::Vector3f& intersec_point);
	bool pick_vertex(int& output_index, int& vertex_index);
	int pick_vertex_per_core(Eigen::MatrixXd& V, Eigen::MatrixXi& F, int LR);
	void follow_and_mark_selected_faces();
	void UpdateEnergyColors(const int index);
	void update_parameters_for_all_cores();
	void clear_sellected_faces_and_vertices();

	//Basic Methods
	igl::opengl::ViewerData& InputModel();
	igl::opengl::ViewerData& OutputModel(const int index);
	igl::opengl::ViewerCore& InputCore();
	igl::opengl::ViewerCore& OutputCore(const int index);
	std::vector<std::pair<OptimizationOutput&, int>> listOfOutputsToUpdate(const int out_index);

	void change_minimizer_type(Cuda::OptimizerType type);
	void draw_brush_sphere();
	void load_new_model(const std::string modelpath);
	void update_core_settings(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
	void update_data_from_minimizer();
	
	//Start/Stop the minimizer Thread
	void init_objective_functions(const int index);
	void stop_all_minimizers_threads();
	void stop_one_minimizer_thread(const OptimizationOutput o);
	void start_all_minimizers_threads();
	void start_one_minimizer_thread(const OptimizationOutput o);
	bool is_Any_Minizer_running();
	void run_one_minimizer_iter();
	void init_aux_variables();

	//FD check
	void checkGradients();

	//outputs
	void add_output();
	void remove_output(const int output_index);
};

#endif