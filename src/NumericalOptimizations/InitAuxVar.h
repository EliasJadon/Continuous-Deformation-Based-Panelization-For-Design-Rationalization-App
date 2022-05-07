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

#define INIT_AUX_VAR_MENU	("SPHERE_AUTO\0"						\
							"SPHERE_AUTO_ALIGNED_TO_NORMAL\0"		\
							"SPHERE_MANUAL_ALIGNED_TO_NORMAL\0"		\
							"SPHERE_AUTO_CENTER_POINT\0"			\
							"CYLINDER_AUTO\0"						\
							"CYLINDER_AUTO_ALIGNED_TO_NORMAL\0"		\
							"CYLINDER_MANUAL_ALIGNED_TO_NORMAL\0"	\
							"CYLINDER_MANUAL_PER_FACE_ALIGNED_TO_NORMAL\0"	\
							"CYLINDER_VECTOR_HELPER_ALIGNED_TO_NORMAL\0\0")

namespace NumericalOptimizations {
	namespace InitAuxVar {
		enum type {
			SPHERE_AUTO,
			SPHERE_AUTO_ALIGNED_TO_NORMAL,
			SPHERE_MANUAL_ALIGNED_TO_NORMAL,
			SPHERE_AUTO_CENTER_POINT,
			CYLINDER_AUTO,
			CYLINDER_AUTO_ALIGNED_TO_NORMAL,
			CYLINDER_MANUAL_ALIGNED_TO_NORMAL,
			CYLINDER_MANUAL_PER_FACE_ALIGNED_TO_NORMAL,
			CYLINDER_VECTOR_HELPER_ALIGNED_TO_NORMAL
		};

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

		void cylinder_fit_wrapper(
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
