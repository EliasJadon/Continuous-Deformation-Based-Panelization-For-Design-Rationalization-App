#pragma once
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include "deformation_plugin.h"

class MeshSimplificationMenu : public igl::opengl::glfw::imgui::ImGuiMenu
{	
public:
	MeshSimplificationMenu(){}
	~MeshSimplificationMenu(){}
	IGL_INLINE virtual void draw_viewer_menu() override {
		deformation_plugin* dp = (deformation_plugin*)plugin;
		if (dp->ui.status != app_utils::UserInterfaceOptions::NONE)
		{
			dp->CollapsingHeader_user_interface();
			dp->Draw_output_window();
			dp->Draw_results_window();
			dp->Draw_energies_window();
			return;
		}
		float w = ImGui::GetContentRegionAvail().x;
		float p = ImGui::GetStyle().FramePadding.x;
		
		if (ImGui::Button("save##mesh"))
			viewer->open_dialog_save_mesh();

		ImGui::Combo("active output", (int*)(&dp->ActiveOutput), app_utils::build_outputs_list(dp->Outputs.size()));

		if (ImGui::Button("save sphere", ImVec2((w - p) / 2.f, 0)) && dp->Outputs[dp->ActiveOutput].clustering_faces_indices.size()) {
			// multiply all the mesh by "factor". relevant only for spheres. 
			double factor = 1;
			for (auto& obj : dp->Outputs[dp->ActiveOutput].totalObjective->objectiveList) {
				auto fr = std::dynamic_pointer_cast<fixRadius>(obj);
				if (fr != NULL && fr->w != 0)
					factor = fr->alpha;
			}
			// get mesh data
			OptimizationOutput o = dp->Outputs[dp->ActiveOutput];
			Eigen::MatrixX3d colors = o.clustering_faces_colors;
			Eigen::MatrixXd v_out = factor * dp->OutputModel(dp->ActiveOutput).V;
			Eigen::MatrixXd v_in = factor * dp->InputModel().V;
			Eigen::MatrixXi f = dp->OutputModel(dp->ActiveOutput).F;
			Eigen::VectorXd radiuses = factor * dp->Outputs[dp->ActiveOutput].getRadiusOfSphere();
			Eigen::MatrixXd centers = factor * dp->Outputs[dp->ActiveOutput].getCenterOfSphere();

			// create new directory for saving the data
			std::string main_file_path = OptimizationUtils::ProjectPath() + "models\\outputmodels\\" + dp->modelName + app_utils::CurrentTime() + "\\";
			std::string aux_file_path = main_file_path + "auxiliary_variables\\";
			std::string parts_file_path = main_file_path + "sphere_parts\\";
			std::string parts_color_file_path = main_file_path + "sphere_parts_with_colors\\";
			std::string file_name = dp->modelName + std::to_string(dp->ActiveOutput);
			if (mkdir(main_file_path.c_str()) == -1 ||
				mkdir(parts_file_path.c_str()) == -1 ||
				mkdir(aux_file_path.c_str()) == -1 ||
				mkdir(parts_color_file_path.c_str()) == -1)
			{
				std::cerr << "error :  " << strerror(errno) << std::endl;
				exit(1);
			}

			// save each cluster in the new directory
			for (int clus_index = 0; clus_index < o.clustering_faces_indices.size(); clus_index++)
			{
				std::vector<int> clus_faces_index = o.clustering_faces_indices[clus_index];
				Eigen::MatrixX3i clus_faces_val(clus_faces_index.size(), 3);
				Eigen::MatrixX3d clus_faces_color(clus_faces_index.size(), 3);

				double sumradius = 0;
				Eigen::RowVector3d sumcenters(0, 0, 0);
				for (int fi = 0; fi < clus_faces_index.size(); fi++)
				{
					sumradius += radiuses(clus_faces_index[fi]);
					sumcenters += centers.row(clus_faces_index[fi]);
					clus_faces_val.row(fi) = f.row(clus_faces_index[fi]);
					clus_faces_color.row(fi) = colors.row(clus_faces_index[fi]);
				}
				Eigen::RowVector3d avgcenter = sumcenters / clus_faces_index.size();
				double avgradius = sumradius / clus_faces_index.size();

				Eigen::MatrixX3d clus_vertices(v_out.rows(), 3);
				for (int vi = 0; vi < v_out.rows(); vi++)
					clus_vertices.row(vi) = v_out.row(vi);
				// save the current cluster in "off" file format
				std::string clus_file_name = parts_file_path + file_name + "_sphere_" + std::to_string(clus_index) + ".off";
				std::string clus_file_name_colors = parts_color_file_path + file_name + "_sphere_" + std::to_string(clus_index) + "_withcolors.off";
				app_utils::writeOFFwithColors(clus_file_name_colors, clus_vertices, clus_faces_val, clus_faces_color);
				igl::writeOFF(clus_file_name, clus_vertices, clus_faces_val);
			}
			// save the final mesh in "off" file format
			igl::writeOFF(main_file_path + file_name + "_output.off", v_out, f);
			igl::writeOFF(main_file_path + file_name + "_input.off", v_in, f);
			app_utils::writeOFFwithColors(main_file_path + file_name + "_output_withcolors.off", v_out, f, colors);
			app_utils::writeOFFwithColors(main_file_path + file_name + "_input_withcolors.off", v_in, f, colors);
			app_utils::writeTXTFile(main_file_path + file_name + "readme.txt", dp->modelName, true,
				o.clustering_faces_indices, v_out, f, colors, radiuses, centers);
			app_utils::write_txt_sphere_fabrication_file(main_file_path + file_name + "fabrication.txt",
				o.clustering_faces_indices, v_out, f, radiuses, centers);
			//save auxiliary variables
			Eigen::MatrixXi temp(1, 3);
			temp << 1, 3, 2;
			igl::writeOFF(aux_file_path + file_name + "_aux_centers.off", centers, temp);
			Eigen::MatrixXd mat_radiuses(radiuses.size(), 3);
			mat_radiuses.setZero();
			mat_radiuses.col(0) = radiuses;
			igl::writeOFF(aux_file_path + file_name + "_aux_radiuses.off", mat_radiuses, temp);
		}
		ImGui::SameLine();
		if (ImGui::Button("save planar", ImVec2((w - p) / 2.f, 0)) && dp->Outputs[dp->ActiveOutput].clustering_faces_indices.size()) {
			// get mesh data
			OptimizationOutput o = dp->Outputs[dp->ActiveOutput];
			Eigen::MatrixX3d colors = o.clustering_faces_colors;
			Eigen::MatrixXd v_out = dp->OutputModel(dp->ActiveOutput).V;
			Eigen::MatrixXd v_in = dp->InputModel().V;
			Eigen::MatrixXi f = dp->OutputModel(dp->ActiveOutput).F;
			Eigen::VectorXd radiuses = dp->Outputs[dp->ActiveOutput].getRadiusOfSphere();
			Eigen::MatrixXd centers = dp->Outputs[dp->ActiveOutput].getCenterOfSphere();
			Eigen::MatrixXd normals = dp->Outputs[dp->ActiveOutput].getFacesNormals();

			// create new directory for saving the data
			std::string main_file_path = OptimizationUtils::ProjectPath() + "models\\outputmodels\\" + dp->modelName + app_utils::CurrentTime() + "\\";
			std::string aux_file_path = main_file_path + "auxiliary_variables\\";
			std::string parts_file_path = main_file_path + "polygon_parts\\";
			std::string parts_color_file_path = main_file_path + "polygon_parts_with_colors\\";
			std::string file_name = dp->modelName + std::to_string(dp->ActiveOutput);
			if (mkdir(main_file_path.c_str()) == -1 ||
				mkdir(parts_file_path.c_str()) == -1 ||
				mkdir(aux_file_path.c_str()) == -1 ||
				mkdir(parts_color_file_path.c_str()) == -1)
			{
				std::cerr << "error :  " << strerror(errno) << std::endl;
				exit(1);
			}

			// save each cluster in the new directory
			for (int polygon_index = 0; polygon_index < o.clustering_faces_indices.size(); polygon_index++)
			{
				std::vector<int> clus_f_indices = o.clustering_faces_indices[polygon_index];
				const int clus_num_faces = clus_f_indices.size();
				Eigen::MatrixX3i clus_f(clus_num_faces, 3);
				Eigen::MatrixX3d clus_color(clus_num_faces, 3);

				for (int fi = 0; fi < clus_num_faces; fi++)
				{
					clus_f.row(fi) = f.row(clus_f_indices[fi]);
					clus_color.row(fi) = colors.row(clus_f_indices[fi]);
				}
				// save the current cluster in "off" file format
				std::string clus_file_name = parts_file_path + file_name + "_polygon_" + std::to_string(polygon_index) + ".off";
				std::string clus_file_name_colors = parts_color_file_path + file_name + "_polygon_" + std::to_string(polygon_index) + "_withcolors.off";
				igl::writeOFF(clus_file_name, v_out, clus_f);
				app_utils::writeOFFwithColors(clus_file_name_colors, v_out, clus_f, clus_color);
			}
			// save the final mesh in "off" file format
			igl::writeOFF(main_file_path + file_name + "_input.off", v_in, f);
			igl::writeOFF(main_file_path + file_name + "_output.off", v_out, f);
			app_utils::writeOFFwithColors(main_file_path + file_name + "_input_withcolors.off", v_in, f, colors);
			app_utils::writeOFFwithColors(main_file_path + file_name + "_output_withcolors.off", v_out, f, colors);
			app_utils::writeTXTFile(main_file_path + file_name + "readme.txt", dp->modelName, false,
				o.clustering_faces_indices, v_out, f, colors, radiuses, centers);
			//save auxiliary variables
			Eigen::MatrixXi temp(1, 3);
			temp << 1, 3, 2;
			igl::writeOFF(aux_file_path + file_name + "_aux_normals.off", normals, temp);
		}

		ImGui::Checkbox("outputs window", &dp->outputs_window);
		ImGui::Checkbox("results window", &dp->results_window);
		ImGui::Checkbox("energy window", &dp->energies_window);
		dp->CollapsingHeader_face_coloring();
		dp->CollapsingHeader_screen();
		dp->CollapsingHeader_measures();
		dp->CollapsingHeader_clustering();
		dp->CollapsingHeader_fabrication();
		dp->CollapsingHeader_minimizer();
		dp->CollapsingHeader_cores(viewer->core(dp->inputCoreID), viewer->data(dp->inputModelID));
		dp->CollapsingHeader_models(viewer->data(dp->inputModelID));
		dp->CollapsingHeader_colors();
		dp->Draw_output_window();
		dp->Draw_results_window();
		dp->Draw_energies_window();
		dp->CollapsingHeader_update();
	}
};

