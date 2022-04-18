#include <igl/opengl/glfw/Viewer.h>
#include "deformation_plugin.h"
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include "FabricationMenu.h"

int main(int argc, char *argv[])
{
	FabricationMenu menu;
	menu.name = "Fabrication menu";

	deformation_plugin plugin;
	plugin.plugin_name = "Fabrication Plugin";
	plugin.widgets.push_back(&menu);

	igl::opengl::glfw::Viewer viewer;
	viewer.plugins.push_back(&plugin);
	viewer.launch(/*resizable*/ true, /*fullscreen*/ true, "Fabrication viewer");

	return EXIT_SUCCESS;
}


