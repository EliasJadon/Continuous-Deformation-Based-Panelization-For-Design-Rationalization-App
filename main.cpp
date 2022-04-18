#include <igl/opengl/glfw/Viewer.h>
#include "deformation_plugin.h"
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>

int main(int argc, char *argv[])
{
  igl::opengl::glfw::Viewer viewer;
  deformation_plugin plugin;
  viewer.plugins.push_back(&plugin);
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  plugin.widgets.push_back(&menu);
  viewer.launch();
  return EXIT_SUCCESS;
}


