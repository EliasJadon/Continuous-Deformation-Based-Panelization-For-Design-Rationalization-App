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

		// A wrapper function which loop over all faces
		// and find for each face the most suitable sphere
		void sphere_fit_wrapper(
			const int adjacency_level,
			const Eigen::MatrixXd& V,
			const Eigen::MatrixXi& F,
			Eigen::MatrixXd& C,
			Eigen::VectorXd& R);

		// A wrapper function which loop over all faces
		// and find for each face the most suitable sphere 
		// which is aligned to normal
		void sphere_fit_aligned_to_normal_wrapper(
			const int adjacency_level,
			const Eigen::MatrixXd& V,
			const Eigen::MatrixXi& F,
			Eigen::MatrixXd& C,
			Eigen::VectorXd& R);

		void Least_Squares_Cylinder_Fit(
			const int imax,
			const int jmax,
			const int Distance,
			const Eigen::MatrixXd& V,
			const Eigen::MatrixXi& F,
			Eigen::MatrixXd& center0,
			Eigen::MatrixXd& dir0,
			Eigen::VectorXd& radius0);

	};


};
