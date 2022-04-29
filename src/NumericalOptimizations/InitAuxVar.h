#pragma once

#include <direct.h>
#include <iostream>
#include <igl/doublearea.h>
#include <igl/local_basis.h>
#include <igl/boundary_loop.h>
#include <igl/per_face_normals.h>
#include <windows.h>
#include <Eigen/sparse>
#include <igl/vertex_triangle_adjacency.h>
#include <chrono>
#include <igl/triangle_triangle_adjacency.h>
#include <set>
#include <igl/PI.h>

namespace NumericalOptimizations {
	namespace InitAuxVar {
		void general_sphere_fit(
			const int Distance_from,
			const int Distance_to,
			const Eigen::MatrixXd& V,
			const Eigen::MatrixXi& F,
			Eigen::MatrixXd& center0,
			Eigen::VectorXd& radius0);
	};
};
