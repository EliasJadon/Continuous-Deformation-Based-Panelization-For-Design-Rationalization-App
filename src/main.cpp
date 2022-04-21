#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/file_dialog_open.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include "GUIExtensions/MeshSimplificationPlugin.h"
#include "GUIExtensions/MeshSimplificationMenu.h"


static int extract_mesh_name_from_path(std::string& mesh_name, const std::string& mesh_path)
{
	size_t head = mesh_path.rfind('\\');
	size_t tail = mesh_path.rfind('.');
	if ((head == std::string::npos) || (tail == std::string::npos)) {
		std::cerr << "Error: No file name found in: " << mesh_path << std::endl;
		return EXIT_FAILURE;
	}
	mesh_name = mesh_path.substr(head + 1, tail - head - 1);
	return EXIT_SUCCESS;
}

static int extract_mesh_extension_from_path(std::string& mesh_extension, const std::string& mesh_path)
{
	size_t last_dot = mesh_path.rfind('.');
	if (last_dot == std::string::npos) {
		std::cerr << "Error: No file extension found in: " << mesh_path << std::endl;
		return EXIT_FAILURE;
	}
	mesh_extension = mesh_path.substr(last_dot + 1);
	return EXIT_SUCCESS;
}

int main()
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	std::string mesh_path, mesh_name, mesh_extension;

	// Get mesh file from user
	mesh_path = igl::file_dialog_open();
	if (extract_mesh_name_from_path(mesh_name, mesh_path))
		return EXIT_FAILURE;
	if (extract_mesh_extension_from_path(mesh_extension, mesh_path))
		return EXIT_FAILURE;

	// Read mesh vertices & faces
	if (mesh_extension == "off" || mesh_extension == "OFF") {
		if (!igl::readOFF(mesh_path, V, F)) {
			std::cerr << "Error: readOFF failed!" << std::endl;
			return EXIT_FAILURE;
		}
	}
	else if (mesh_extension == "obj" || mesh_extension == "OBJ") {
		if (!igl::readOBJ(mesh_path, V, F)) {
			std::cerr << "Error: readOBJ failed!" << std::endl;
			return EXIT_FAILURE;
		}
	}
	else if (mesh_extension == "ply" || mesh_extension == "PLY") {
		if (!igl::readPLY(mesh_path, V, F)) {
			std::cerr << "Error: readPLY failed!" << std::endl;
			return EXIT_FAILURE;
		}
	}
	else {
		std::cerr << "Error! the following mesh extension is not supported: " << mesh_extension << std::endl;
		return EXIT_FAILURE;
	}

	if (V.rows() < 3 || V.cols() != 3) {
		std::cerr << "Error: V format is illegal!" << std::endl;
		return EXIT_FAILURE;
	}
	if (F.rows() < 1 || F.cols() != 3) {
		std::cerr << "Error: F format is illegal!" << std::endl;
		return EXIT_FAILURE;
	}

	GUIExtensions::MeshSimplificationMenu menu;
	menu.name = "Mesh Simplification menu";

	GUIExtensions::MeshSimplificationPlugin plugin;
	plugin.plugin_name = "Mesh Simplification Plugin";
	plugin.widgets.push_back(&menu);
	plugin.original_F = F;
	plugin.original_V = V;
	plugin.modelName = mesh_name;

	igl::opengl::glfw::Viewer viewer;
	viewer.plugins.push_back(&plugin);
	viewer.launch(/*resizable*/ true, /*fullscreen*/ false, "Mesh Simplification App");

	return EXIT_SUCCESS;
}


