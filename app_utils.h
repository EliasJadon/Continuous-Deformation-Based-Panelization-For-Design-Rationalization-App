#pragma once

#include <iostream>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/unproject_in_mesh.h>
#include <igl/unproject_ray.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <igl/adjacency_matrix.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/edge_lengths.h>
#include <igl/boundary_loop.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui.h>
#include <chrono>
#include <vector>
#include <queue>
#include "unique_colors.h"

#include "Minimizer.h"
#include "STVK.h"
#include "SDenergy.h"
#include "FixAllVertices.h"
#include "AuxBendingNormal.h"
#include "AuxSpherePerHinge.h"
#include "FixChosenConstraints.h"
#include "fixRadius.h"
#include "UniformSmoothness.h"
#include "ClusterHard.h"
#include "BendingNormal.h"

#define RED_COLOR Eigen::Vector3f(1, 0, 0)
#define BLUE_COLOR Eigen::Vector3f(0, 0, 1)
#define GREEN_COLOR Eigen::Vector3f(0, 1, 0)
#define GOLD_COLOR Eigen::Vector3f(1, 215.0f / 255.0f, 0)
#define GREY_COLOR Eigen::Vector3f(0.75, 0.75, 0.75)
#define WHITE_COLOR Eigen::Vector3f(1, 1, 1)
#define BLACK_COLOR Eigen::Vector3f(0, 0, 0)
#define M_PI 3.14159

namespace app_utils {
	enum Face_Colors { 
		NO_COLORS, 
		NORMALS_CLUSTERING, 
		SPHERES_CLUSTERING,
		SIGMOID_PARAMETER
	};
	enum View {
		HORIZONTAL = 0,
		VERTICAL,
		SHOW_INPUT_SCREEN_ONLY,
		SHOW_OUTPUT_SCREEN_ONLY_0
	};
	enum Neighbor_Type {
		CURR_FACE,
		LOCAL_SPHERE,
		GLOBAL_SPHERE,
		LOCAL_NORMALS,
		GLOBAL_NORMALS
	};
	enum UserInterfaceOptions { 
		NONE,
		FIX_VERTICES,
		FIX_FACES,
		BRUSH_WEIGHTS_INCR,
		BRUSH_WEIGHTS_DECR,
		ADJ_WEIGHTS
	};
	enum ClusteringType { NO_CLUS, Circle_clustering, Agglomerative_hierarchical, External };
	
	static bool writeOFFwithColors(
		const std::string& path,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F,
		const Eigen::MatrixXd& C)
	{
		if (V.cols() != 3 || F.cols() != 3 || C.cols() != 3)
			return false;
		if (V.rows() <= 0 || F.rows() <= 0 || F.rows() != C.rows())
			return false;

		std::ofstream myfile;
		myfile.open(path);
		myfile << "OFF\n";
		myfile << V.rows() << " " << F.rows() << " 0\n";
		for (int vi = 0; vi < V.rows(); vi++) {
			myfile << V(vi, 0) << " " << V(vi, 1) << " " << V(vi, 2) << "\n";
		}
		for (int fi = 0; fi < F.rows(); fi++) {
			myfile << "3 " << F(fi, 0) << " " << F(fi, 1) << " " << F(fi, 2) << " ";
			myfile << int(255 * C(fi, 0)) << " " << int(255 * C(fi, 1)) << " " << int(255 * C(fi, 2)) << "\n";
		}
		myfile.close();
		return true;
	}

	static void readPLY2(
		const std::string& path,
		Eigen::MatrixXd& V,
		Eigen::MatrixX3i& F,
		Eigen::VectorXi& Clus)
	{
		std::ifstream myfile(path.c_str());
		{
			//Data integrity checkpoint
			std::size_t found = path.find_last_of(".");
			if (found == std::string::npos || (path.substr(found + 1)).compare("ply2") != 0) {
				std::cout << "Error! This is not a PLY2 file: " << path << std::endl;
				exit(1);
			}
			if (!(myfile.is_open())) {
				std::cout << "Error! couldn't open the file: " << path << std::endl;
				exit(1);
			}
		}
		int numV,numF;
		myfile >> numV >> numF;
		if (numV < 3 || numF < 1) {
			std::cout << "Error! numV = " << numV << ", numF = " << numF << std::endl;
			exit(1);
		}
		V.resize(numV, 3);
		F.resize(numF, 3);
		Clus.resize(numF);
		for (int i = 0; i < numV; i++)
			myfile >> V(i, 0) >> V(i, 1) >> V(i, 2);
		for (int i = 0; i < numF; i++) {
			int vertices_each_face;
			myfile >> vertices_each_face;
			if (vertices_each_face != 3) {
				std::cout << "Error! each face should contain 3 vertices not " << vertices_each_face << std::endl;
				exit(1);
			}
			myfile >> F(i, 0) >> F(i, 1) >> F(i, 2) >> Clus(i);
		}
		myfile.close();
	}

	static std::vector<std::vector<int>> get_Clusters_Vector_From_Matrix(const Eigen::VectorXi& Clus) {
		const int num_clusters = Clus.maxCoeff() + 1;
		const int num_faces = Clus.size();
		std::vector<std::vector<int>> c(num_clusters);
		for (int fi = 0; fi < num_faces; fi++)
			c[Clus[fi]].push_back(fi);
		return c;
	}

	static std::vector<int> findPathVertices_usingDFS(
		const std::vector<std::vector<int> >& A, 
		const int s, 
		const int target,
		const int numV) 
	{
		if (s < 0 || target < 0 || s >= numV || target >= numV)
			return {};
		if (s == target)
			return { s };

		std::vector<bool> seen(numV, false);
		std::vector<std::vector<std::pair<int, int>>> Vertices_per_level;
		Vertices_per_level.push_back({ {s,-1} });
		seen[s] = true;
		bool finish = false;
		while (!finish) {
			std::vector<std::pair<int, int>> currV;
			const int level = Vertices_per_level.size() - 1;
			for (std::pair<int, int> new_s : Vertices_per_level[level]) {
				for (int neighbour : A[new_s.first]) {
					if (neighbour == target) {
						finish = true;
					}
					if (!seen[neighbour]) {
						currV.push_back({ neighbour,new_s.first });
						seen[neighbour] = true;
					}
				}
			}
			Vertices_per_level.push_back(currV);
		}

		//Part 2: find vertices path
		std::vector<int> path;
		path.push_back(target);
		int last_level = Vertices_per_level.size() - 1;
		while (last_level > 0) {
			int v_target = path[path.size() - 1];
			for (auto& v : Vertices_per_level[last_level])
				if (v.first == v_target) {
					path.push_back(v.second);
				}
			last_level--;
		}
		return path;
	}

	static int calculate_2_spheres_intersection(
		Eigen::RowVector3d C, const double R,
		Eigen::RowVector3d c, const double r,
		double* compass_length,
		Eigen::RowVector3d* compass_point,
		double* inter_r)
	{
		// See https://mathworld.wolfram.com/Sphere-SphereIntersection.html
		// Now, we translate the spheres to:
		// C_new = (0,0,0) , c_new = (d,0,0)
		// The radius still R & r
		double d = (c - C).norm();
		if (d >= (r + R))
			return 0;
		double x = (pow(d, 2) - pow(r, 2) + pow(R, 2)) / (2 * d);
		double a = sqrt(pow(R, 2) - pow(x, 2));
		double h1 = R - fabs(x);
		*compass_length = sqrt(pow(h1, 2) + pow(a, 2));
		*inter_r = a;
		if (x > 0)
			* compass_point = ((c - C).normalized()) * R;
		else
			*compass_point = ((C - c).normalized()) * R;
		return 1;
	}

	static void sphere_fabrication_data_into_txt_file(
		const std::string& path,
		const std::vector<Eigen::RowVector3d> c, 
		const std::vector<double> r)
	{
		const int num_spheres = c.size();
		if (c.size() != r.size()) {
			printf("Error! the sizes are not equal (%d != %d).\n", c.size(), r.size());
			exit(1);
		}
		FILE* f = fopen(path.c_str(), "w");
		if (f == NULL)
		{
			printf("Error opening file!\n");
			exit(1);
		}
		fprintf(f, "****************** Input Data ******************\n");
		for (int i = 0; i < num_spheres; i++)
			fprintf(f, "Sphere %2d: %7.2f, (%7.2f, %7.2f, %7.2f)\n", i, r[i], c[i][0], c[i][1], c[i][2]);
		fprintf(f, "************************************************\n\n\n");

		fprintf(f, "************************** Intersections **************************\n");
		for (int i = 0; i < num_spheres; i++) {
			fprintf(f, "\n\n\nSphere %2d:\n", i);
			std::vector<std::pair<Eigen::RowVector3d, int>> compass_points;
			for (int j = 0; j < num_spheres; j++) {
				double compass_length, inter_r;
				Eigen::RowVector3d compass_point;
				if (i != j && calculate_2_spheres_intersection(
					c[i], r[i], c[j], r[j], &compass_length, &compass_point, &inter_r))
				{
					fprintf(f, "\tsphere %2d: compassLength=%7.2f, (For sanity test: radius=%7.2f)\n", j, compass_length, inter_r);
					compass_points.push_back(std::pair<Eigen::RowVector3d, int>(compass_point, j));
				}
			}
			for (int k1 = 0; k1 < compass_points.size(); k1++) {
				for (int k2 = 0; k2 < compass_points.size(); k2++) {
					fprintf(f, "\t\tDis(%2d,%2d)=%7.2f\n", compass_points[k1].second, compass_points[k2].second,
						(compass_points[k1].first - compass_points[k2].first).norm());
				}
			}
		}
		fprintf(f, "\n******************************************************************\n\n\n");
		fclose(f);
	}

	static bool write_txt_sphere_fabrication_file(
		const std::string& path,
		const std::vector<std::vector<int>>& clustering_faces_indices,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F,
		const Eigen::VectorXd& Radiuses,
		const Eigen::MatrixX3d& Centers)
	{
		if (V.cols() != 3 || F.cols() != 3 || V.rows() <= 0 || F.rows() <= 0)
			return false;
		std::vector<Eigen::RowVector3d> c;
		std::vector<double> r;
		for (int ci = 0; ci < clustering_faces_indices.size(); ci++) {
			double avgRadius = 0;
			Eigen::RowVector3d avgCenter(0, 0, 0);
			for (int fi = 0; fi < clustering_faces_indices[ci].size(); fi++) {
				const int face_index = clustering_faces_indices[ci][fi];
				avgRadius += Radiuses[face_index];
				avgCenter = avgCenter + Centers.row(face_index);
			}
			avgRadius /= clustering_faces_indices[ci].size();
			avgCenter /= clustering_faces_indices[ci].size();
			r.push_back(avgRadius);
			c.push_back(avgCenter);
		}
		sphere_fabrication_data_into_txt_file(path, c, r);
		return true;
	}
	
	static std::string CurrentTime() {
		char date_buffer[80] = { 0 };
		{
			time_t rawtime_;
			struct tm* timeinfo_;
			time(&rawtime_);
			timeinfo_ = localtime(&rawtime_);
			strftime(date_buffer, 80, "_%H_%M_%S__%d_%m_%Y", timeinfo_);
		}
		return std::string(date_buffer);
	}

	static bool writeTXTFile(
		const std::string& path,
		const std::string& modelName,
		const bool isSphere,
		const std::vector<std::vector<int>>& clustering_faces_indices,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F,
		const Eigen::MatrixXd& C,
		const Eigen::VectorXd& Radiuses,
		const Eigen::MatrixX3d& Centers)
	{
		if (V.cols() != 3 || F.cols() != 3 || C.cols() != 3)
			return false;
		if (V.rows() <= 0 || F.rows() <= 0 || F.rows() != C.rows())
			return false;


		std::ofstream myfile;
		myfile.open(path);
		myfile << "\n\n===============================================\n";
		myfile << "Model name: \t"						<< modelName << "\n";
		myfile << "Num Faces: \t"						<< F.rows() << "\n";
		myfile << "Num Vertices: \t"					<< V.rows() << "\n";
		if (isSphere) {
			myfile << "Num spheres: \t" << clustering_faces_indices.size() << "\n";
		}
		else {
			myfile << "Num polygons: \t" << clustering_faces_indices.size() << "\n";
			myfile << "-----------------------List of polygons:" << "\n";
		}
		myfile << "===============================================\n\n\n";
		
		for (int ci = 0; ci < clustering_faces_indices.size(); ci++) {
			myfile << "\n";
			//calculating the avg center&radius for each group/cluster
			double avgRadius = 0;
			Eigen::RowVector3d avgCenter(0, 0, 0), avgColor(0, 0, 0);
			for (int fi = 0; fi < clustering_faces_indices[ci].size(); fi++) {
				const int face_index = clustering_faces_indices[ci][fi];
				if (isSphere) {
					avgRadius += Radiuses[face_index];
					avgCenter = avgCenter + Centers.row(face_index);
				}
				avgColor = avgColor + C.row(face_index);
			}
			if (isSphere) {
				avgRadius /= clustering_faces_indices[ci].size();
				avgCenter /= clustering_faces_indices[ci].size();
			}
			avgColor /= clustering_faces_indices[ci].size();
			

			//output data
			if (isSphere) {
				myfile << "Sphere ID:\t" << ci << "\n";
				myfile << "Radius length: " << avgRadius << "\n";
				myfile << "Center point: " << "(" << avgCenter(0) << ", " << avgCenter(1) << ", " << avgCenter(2) << ")" << "\n";
			}
			else {
				myfile << "Polygon ID:\t" << ci << "\n";
			}
			
			myfile << "color: " << "(" << avgColor(0) << ", " << avgColor(1) << ", " << avgColor(2) << ")" << "\n";
			myfile << "Num faces: " << clustering_faces_indices[ci].size() << "\n";
			myfile << "faces list: ";
			for (int fi = 0; fi < clustering_faces_indices[ci].size(); fi++) {
				const int face_index = clustering_faces_indices[ci][fi];
				myfile << face_index << ", ";
			}
			myfile << "\n";
			myfile << "----------------------------\n";
		}
		
		myfile.close();
		return true;
	}

	static double Hausdorff_distance_one_side(const Eigen::MatrixXd& V1, const Eigen::MatrixXd& V2)
	{
		double cmax = 0;
		for (int x = 0; x < V1.rows(); x++) {
			double cmin = std::numeric_limits<double>::max();
			for (int y = 0; y < V2.rows(); y++) {
				double d = (V1.row(x) - V2.row(y)).norm();
				cmin = std::min<double>(cmin, d);
			}
			cmax = std::max<double>(cmax, cmin);
		}
		return cmax;
	}

	static double mesh_diameter(const Eigen::MatrixXd& V)
	{
		if (V.cols() != 3) {
			printf("Diameter function Error! Num columns must be 3.\n");
			exit(1);
		}
		double max = 0;
		for (int i = 0; i < V.rows(); i++) {
			for (int j = i+1; j < V.rows(); j++) {
				double d = (V.row(i) - V.row(j)).norm();
				max = (max > d) ? max : d;
			}
		}
		return max;
	}

	static double calculate_sphere_intersection(
		const Eigen::RowVector3d C, const double R,
		const Eigen::RowVector3d c, const double r,
		double* compass_length,
		Eigen::RowVector3d* compass_point,
		double* inter_r) 
	{
		double d = (c - C).norm();

		if (d >= (r + R)) {
			printf("Error! the circles doesn't intersect.\n");
			exit(1);
		}
		if (d > R) {

		}

	}

	static double calculate_sphere_intersection_case1(
		const Eigen::RowVector3d C, const double R,
		const Eigen::RowVector3d c, const double r,
		double* compass_length,
		Eigen::RowVector3d* compass_point,
		double* inter_r)
	{
		// See https://mathworld.wolfram.com/Sphere-SphereIntersection.html
		//Now, we translate the spheres to: 
		// C_new = (0,0,0) , c_new = (d,0,0)
		// The radius still R & r
		double d = (c - C).norm();
		double x = (pow(d, 2) - pow(r, 2) + pow(R, 2)) / (2*d);
		double a = sqrt(pow(R, 2) - pow(x, 2));
		double h1 = R - x;
		*compass_length = sqrt(pow(h1, 2) + pow(a, 2));
		// a is the radius for the intersection circle
		*inter_r = a;
		*compass_point = ((c - C).normalized()) * R;
	}

	static void get_distribution(
		const std::vector<double>& vec,
		double& max,
		double& min,
		double& avg,
		double& std)
	{
		max = std::numeric_limits<double>::min();
		min = std::numeric_limits<double>::max();

		// Data integrity
		for (double val : vec) {
			if (val < 0) {
				printf("Invalid Inpit in get_distribution function!\n");
				exit(1);
			}
		}

		// calculate minimum, maximum and average values
		double sum = 0;
		for (double val : vec) {
			sum += val;
			max = std::max<double>(max, val);
			min = std::min<double>(min, val);
		}
		avg = sum / vec.size();

		//calculate Standard deviation
		sum = 0;
		for (double val : vec)
			sum += pow(val - avg, 2);
		std = sum / vec.size();
	}

	static void planar_error_distribution_1cluster(
		const Eigen::MatrixXd& V, 
		const Eigen::MatrixXi& F, 
		const Eigen::MatrixX3d& N,
		const std::vector<int>& clus,
		std::vector<double>& MSEs)
	{
		for (int i1 = 0; i1 < clus.size(); i1++) {
			for (int i2 = i1 + 1; i2 < clus.size(); i2++) {
				int f1 = clus[i1];
				int f2 = clus[i2];
				double mse_1 = (N.row(f1) - N.row(f2)).squaredNorm();
				double mse_2 = (N.row(f1) + N.row(f2)).squaredNorm();
				double mse = std::min<double>(mse_1, mse_2);
				MSEs.push_back(mse);
			}
		}
	}

	static void sphere_error_distribution_1cluster(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixXd& C,
		const std::vector<int>& clus,
		std::vector<double>& MSEs)
	{
		// Get average center of the current cluster (sphere)
		Eigen::RowVector3d c_avg(0, 0, 0);
		for (int fi : clus)
			c_avg = c_avg + C.row(fi);
		c_avg /= clus.size();

		// Get the vertices of the sphere
		Eigen::VectorXd is_clus_vertices(V.rows());
		is_clus_vertices.setZero();
		for (int fi : clus) {
			is_clus_vertices[F(fi, 0)] = 1;
			is_clus_vertices[F(fi, 1)] = 1;
			is_clus_vertices[F(fi, 2)] = 1;
		}

		// Loop over the **cluster** vertices
		for (int v1 = 0; v1 < V.rows(); v1++) {
			for (int v2 = v1 + 1; v2 < V.rows(); v2++) {
				if (is_clus_vertices[v1] && is_clus_vertices[v2]) {
					double r1 = (V.row(v1) - c_avg).norm();
					double r2 = (V.row(v2) - c_avg).norm();
					double mse = pow(r1 - r2, 2);
					MSEs.push_back(mse);
				}
			}
		}
	}

	static void planar_error_distribution(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const std::vector<std::vector<int>>& clusters,
		double& max,
		double& min,
		double& avg,
		double& std)
	{
		std::vector<double> MSEs; MSEs.clear();
		Eigen::MatrixX3d N;
		igl::per_face_normals(V, F, N);
		for (std::vector<int> clus : clusters)
			planar_error_distribution_1cluster(V, F, N, clus, MSEs);
		get_distribution(MSEs, max, min, avg, std);
	}

	static void sphere_error_distribution(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const Eigen::MatrixXd& C,
		const std::vector<std::vector<int>>& clusters,
		double& max,
		double& min,
		double& avg,
		double& std)
	{
		std::vector<double> MSEs; MSEs.clear();
		for (std::vector<int> clus : clusters)
			sphere_error_distribution_1cluster(V, F, C, clus, MSEs);
		get_distribution(MSEs, max, min, avg, std);
	}
	
	static double diagonal_of_bounding_box(const Eigen::MatrixXd& V)
	{
		if (V.cols() != 3) {
			printf("diagonal_of_bounding_box function Error! Num columns must be 3.\n");
			exit(1);
		}
		double x_max = std::numeric_limits<double>::min();
		double y_max = std::numeric_limits<double>::min();
		double z_max = std::numeric_limits<double>::min();
		double x_min = std::numeric_limits<double>::max();
		double y_min = std::numeric_limits<double>::max();
		double z_min = std::numeric_limits<double>::max();
		for (int i = 0; i < V.rows(); i++) {
			x_max = std::max<double>(V(i, 0), x_max);
			y_max = std::max<double>(V(i, 1), y_max);
			z_max = std::max<double>(V(i, 2), z_max);
			x_min = std::min<double>(V(i, 0), x_min);
			y_min = std::min<double>(V(i, 1), y_min);
			z_min = std::min<double>(V(i, 2), z_min);
		}
		Eigen::Vector3d max_p(x_max, y_max, z_max);
		Eigen::Vector3d min_p(x_min, y_min, z_min);
		return (max_p - min_p).norm();
	}

	static double Hausdorff_distance(const Eigen::MatrixXd& V1, const Eigen::MatrixXd& V2)
	{
		// For more details:
		// https://content.iospress.com/articles/integrated-computer-aided-engineering/ica544
		double d1 = Hausdorff_distance_one_side(V1, V2);
		double d2 = Hausdorff_distance_one_side(V2, V1);
		return std::max<double>(d1, d2);
	}

	static void getPlanarCoordinates(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const std::vector<int>& clus,
		const float scale) 
	{
		//get the boundary vertices
		Eigen::MatrixXi clus_F(clus.size(), 3);
		for (int i = 0; i < clus.size(); i++) {
			clus_F.row(i) = F.row(clus[i]);
		}
		Eigen::VectorXi bnd;
		igl::boundary_loop(clus_F, bnd);

		//get the best-plane edges
		Eigen::RowVector3d argminE1, argminE2;
		double min = std::numeric_limits<double>::max();
		for (int v1 = 1; v1 < bnd.size(); v1++) {
			for (int v2 = v1 + 1; v2 < bnd.size(); v2++) {
				Eigen::RowVector3d e1 = (V.row(bnd[v1]) - V.row(bnd[0])).normalized();
				Eigen::RowVector3d e2 = (V.row(bnd[v2]) - V.row(bnd[0])).normalized();
				double dot2 = pow(e1.dot(e2), 2);
				if (dot2 < min) {
					min = dot2;
					argminE1 = e1;
					argminE2 = e2;
				}
			}
		}

		//get axis X,Y,Z
		Eigen::RowVector3d X_axis = (argminE1).normalized();
		Eigen::RowVector3d Z_axis = (X_axis.cross(argminE2)).normalized();
		Eigen::RowVector3d Y_axis = (X_axis.cross(Z_axis)).normalized();

		//put all vertices on the plane
		std::vector<std::pair<double, double>> plane_coordinates;
		plane_coordinates.clear();
		for (int vi = 0; vi < bnd.size(); vi++) {
			double epsilon = 0.0001;
			Eigen::RowVector3d vec_V = V.row(bnd[vi]) - V.row(bnd[0]);
			double x_coordinate = scale * vec_V.dot(X_axis);
			double y_coordinate = scale * vec_V.dot(Y_axis);
			if (x_coordinate < epsilon && x_coordinate > -epsilon)
				x_coordinate = 0;
			if (y_coordinate < epsilon && y_coordinate > -epsilon)
				y_coordinate = 0;
			plane_coordinates.push_back(std::pair<double, double>(x_coordinate, y_coordinate));
		}

		//print polygon
		std::cout << "Polygon((" << plane_coordinates[0].first
			<< ", " << plane_coordinates[0].second << ")";
		for (int i = 1; i < plane_coordinates.size(); i++) {
			std::cout << ", (" << plane_coordinates[i].first
				<< ", " << plane_coordinates[i].second << ")";
		}
		std::cout << ")\n\n";
	}

	static bool is_cluster_separation_needed_BFS(
		const Eigen::MatrixXi& F, 
		const std::vector<int>& clus,
		std::vector<int>& groupA, 
		std::vector<int>& groupB)
	{
		enum status { NOT_IN_CLUSTER, WAITING, SEEN };
		std::vector<std::vector<std::vector<int>>> TT;
		igl::triangle_triangle_adjacency(F, TT);
		
		std::vector<status> face_status(F.rows());
		for (int fi = 0; fi < face_status.size(); fi++)
			face_status[fi] = NOT_IN_CLUSTER;
		for (const int fi : clus)
			face_status[fi] = WAITING;

		// init...
		std::queue<int> faces_in_current_level;
		faces_in_current_level.push(clus[0]);
		face_status[clus[0]] = SEEN;
		
		// loop using BFS algorithm
		while (faces_in_current_level.size()) {
			const int fi = faces_in_current_level.front();
			faces_in_current_level.pop();
			for (int i = 0; i < 3; i++) {
				if (TT[fi][i].size() && face_status[TT[fi][i][0]] == WAITING) {
					faces_in_current_level.push(TT[fi][i][0]);
					face_status[TT[fi][i][0]] = SEEN;
				}
			}
		}

		// check results
		bool is_split_needed = false;
		groupA.clear(); groupB.clear();
		for (int fi = 0; fi < face_status.size(); fi++) {
			if (face_status[fi] == SEEN)
				groupA.push_back(fi);
			if (face_status[fi] == WAITING) {
				is_split_needed = true;
				groupB.push_back(fi);
			}
		}
		return is_split_needed;
	}

	static bool cluster_separation(
		const Eigen::MatrixXi& F, 
		std::vector<std::vector<int>>& clusters) 
	{
		//init
		std::queue<std::vector<int>> unckecked_clusters;
		for (const std::vector<int> c : clusters)
			unckecked_clusters.push(c);
		clusters.clear();

		// Keep splitting the clusters until there's no need
		bool is_list_updated = false;
		while (unckecked_clusters.size()) {
			const std::vector<int> c = unckecked_clusters.front();
			unckecked_clusters.pop();
			std::vector<int> sub_cluster_A, sub_cluster_B;
			if (is_cluster_separation_needed_BFS(F, c, sub_cluster_A, sub_cluster_B)) {
				is_list_updated = true;
				clusters.push_back(sub_cluster_A);
				unckecked_clusters.push(sub_cluster_B);
			}
			else clusters.push_back(c);
		}
		return is_list_updated;
	}

	static Eigen::RowVector3d computeTranslation(
		const int mouse_x, 
		const int from_x, 
		const int mouse_y, 
		const int from_y, 
		const Eigen::RowVector3d pt3D,
		igl::opengl::ViewerCore& core) 
	{
		Eigen::Matrix4f modelview = core.view;
		//project the given point (typically the handle centroid) to get a screen space depth
		Eigen::Vector3f proj = igl::project(pt3D.transpose().cast<float>().eval(), modelview, core.proj, core.viewport);
		float depth = proj[2];
		double x, y;
		Eigen::Vector3f pos1, pos0;
		//unproject from- and to- points
		x = mouse_x;
		y = core.viewport(3) - mouse_y;
		pos1 = igl::unproject(Eigen::Vector3f(x, y, depth), modelview, core.proj, core.viewport);
		x = from_x;
		y = core.viewport(3) - from_y;
		pos0 = igl::unproject(Eigen::Vector3f(x, y, depth), modelview, core.proj, core.viewport);
		//translation is the vector connecting the two
		Eigen::Vector3f translation;
		translation = pos1 - pos0;
		return Eigen::RowVector3d(translation(0), translation(1), translation(2));
	}

	static int calculateHinges(std::vector<Eigen::Vector2d>& hinges_faceIndex, const Eigen::MatrixX3i& F) {
		std::vector<std::vector<std::vector<int>>> TT;
		igl::triangle_triangle_adjacency(F, TT);
		assert(TT.size() == F.rows());
		hinges_faceIndex.clear();

		///////////////////////////////////////////////////////////
		//Part 1 - Find unique hinges
		for (int fi = 0; fi < TT.size(); fi++) {
			std::vector< std::vector<int>> CurrFace = TT[fi];
			assert(CurrFace.size() == 3 && "Each face should be a triangle (not square for example)!");
			for (std::vector<int> hinge : CurrFace) {
				if (hinge.size() == 1) {
					//add this "hinge"
					int FaceIndex1 = fi;
					int FaceIndex2 = hinge[0];

					if (FaceIndex2 < FaceIndex1) {
						//Skip
						//This hinge already exists!
						//Empty on purpose
					}
					else {
						hinges_faceIndex.push_back(Eigen::Vector2d(FaceIndex1, FaceIndex2));
					}
				}
				else if (hinge.size() == 0) {
					//Skip
					//This triangle has no another adjacent triangle on that edge
					//Empty on purpose
				}
				else {
					//We shouldn't get here!
					//The mesh is invalid
					assert("Each triangle should have only one adjacent triangle on each edge!");
				}

			}
		}
		return hinges_faceIndex.size(); // num_hinges
	}
	
	static std::string ExtractModelName(const std::string& str)
	{
		size_t head, tail;
		head = str.find_last_of("/\\");
		tail = str.find_last_of("/.");
		return (str.substr((head + 1), (tail - head - 1)));
	}
	
	static bool IsMesh2D(const Eigen::MatrixXd& V) {
		return (V.col(2).array() == 0).all();
	}

	static char* build_view_names_list(const int size) {
		std::string cStr("");
		cStr += "Horizontal";
		cStr += '\0';
		cStr += "Vertical";
		cStr += '\0';
		cStr += "InputOnly";
		cStr += '\0';
		for (int i = 0; i < size; i++) {
			std::string sts;
			sts = "OutputOnly " + std::to_string(i);
			cStr += sts.c_str();
			cStr += '\0';
		}
		cStr += '\0';
		int listLength = cStr.length();
		char* comboList = new char[listLength];
		for (unsigned int i = 0; i < listLength; i++)
			comboList[i] = cStr.at(i);
		return comboList;
	}

	static char* build_inputColoring_list(const int size) {
		std::string cStr("");
		cStr += "None";
		cStr += '\0';
		for (int i = 0; i < size; i++) {
			cStr += "Output ";
			cStr += std::to_string(i).c_str();
			cStr += '\0';
		}
		cStr += '\0';
		
		int listLength = cStr.length();
		char* comboList = new char[listLength];
		for (unsigned int i = 0; i < listLength; i++)
			comboList[i] = cStr.at(i);
		
		return comboList;
	}

	static char* build_color_energies_list(const std::shared_ptr<TotalObjective>& totalObjective) {
		std::string cStr("");
		cStr += "No colors";
		cStr += '\0';
		cStr += "Total energy";
		cStr += '\0';
		for (auto& obj : totalObjective->objectiveList) {
			cStr += (obj->name).c_str();
			cStr += '\0';
		}
		cStr += '\0';
		int listLength = cStr.length();
		char* comboList = new char[listLength];
		for (unsigned int i = 0; i < listLength; i++)
			comboList[i] = cStr.at(i);
		return comboList;
	}

	static char* build_outputs_list(const int numOutputs) {
		std::string cStr("");
		for (int i = 0; i < numOutputs; i++) {
			cStr += std::to_string(i);
			cStr += '\0';
		}
		cStr += '\0';
		int listLength = cStr.length();
		char* comboList = new char[listLength];
		for (unsigned int i = 0; i < listLength; i++)
			comboList[i] = cStr.at(i);
		return comboList;
	}

	static Eigen::RowVector3d get_face_avg(
		const igl::opengl::ViewerData& model,
		const int Translate_Index)
	{
		Eigen::RowVector3d avg; avg << 0, 0, 0;
		Eigen::RowVector3i face = model.F.row(Translate_Index);
		avg += model.V.row(face[0]);
		avg += model.V.row(face[1]);
		avg += model.V.row(face[2]);
		avg /= 3;
		return avg;
	}

	static double TriMin(const double v1, const double v2, const double v3) {
		return std::min<double>(std::min<double>(v1, v2), v3);
	}

	static double TriMax(const double v1, const double v2, const double v3) {
		return std::max<double>(std::max<double>(v1, v2), v3);
	}

	static bool grtEqual(const double v1, const double v2, const double epsilon) {
		double diff = v1 - v2;
		if (diff < epsilon && diff > -epsilon)
			return true;
		return diff >= 0;
	}

	static bool lessEqual(const double v1, const double v2, const double epsilon) {
		return grtEqual(v2, v1, epsilon);
	}

	static bool areProjectionsSeparated(
		const double p0,
		const double p1,
		const double p2,
		const double q0,
		const double q1,
		const double q2,
		const double epsilon)
	{
		const double min_p = TriMin(p0, p1, p2);
		const double max_p = TriMax(p0, p1, p2);
		const double min_q = TriMin(q0, q1, q2);
		const double max_q = TriMax(q0, q1, q2);
		return ((grtEqual(min_p, max_q, epsilon)) || (lessEqual(max_p, min_q, epsilon)));
	}

	/**
	 * @function
	 * @param {THREE.Triangle} t1 - Triangular face
	 * @param {THREE.Triangle} t2 - Triangular face
	 * @returns {boolean} Whether the two triangles intersect
	 */
	static bool doTrianglesIntersect(
		const unsigned int t1,
		const unsigned int t2,
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const double epsilon)
	{
		/*
		Adapated from section "4.1 Separation of Triangles" of:
		 - [Dynamic Collision Detection using Oriented Bounding Boxes]
		 (https://www.geometrictools.com/Documentation/DynamicCollisionDetection.pdf)
		*/

		// Triangle 1:
		Eigen::RowVector3d A0 = V.row(F(t1, 0));
		Eigen::RowVector3d A1 = V.row(F(t1, 1));
		Eigen::RowVector3d A2 = V.row(F(t1, 2));
		Eigen::RowVector3d E0 = A1 - A0;
		Eigen::RowVector3d E1 = A2 - A0;
		Eigen::RowVector3d E2 = A2 - A1;
		Eigen::RowVector3d N = E0.cross(E1);
		// Triangle 2:
		Eigen::RowVector3d B0 = V.row(F(t2, 0));
		Eigen::RowVector3d B1 = V.row(F(t2, 1));
		Eigen::RowVector3d B2 = V.row(F(t2, 2));
		Eigen::RowVector3d F0 = B1 - B0;
		Eigen::RowVector3d F1 = B2 - B0;
		Eigen::RowVector3d F2 = B2 - B1;
		Eigen::RowVector3d M = F0.cross(F1);
		Eigen::RowVector3d D = B0 - A0;

		// Only potential separating axes for non-parallel and non-coplanar triangles are tested.
		// Seperating axis: N
		{
			const double p0 = 0;
			const double p1 = 0;
			const double p2 = 0;
			const double q0 = N.dot(D);
			const double q1 = q0 + N.dot(F0);
			const double q2 = q0 + N.dot(F1);
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2, epsilon))
				return false;
		}
		// Separating axis: M
		{
			const double p0 = 0;
			const double p1 = M.dot(E0);
			const double p2 = M.dot(E1);
			const double q0 = M.dot(D);
			const double q1 = q0;
			const double q2 = q0;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2, epsilon))
				return false;
		}
		// Seperating axis: E0 ª F0
		{
			const double p0 = 0;
			const double p1 = 0;
			const double p2 = -(N.dot(F0));
			const double q0 = E0.cross(F0).dot(D);
			const double q1 = q0;
			const double q2 = q0 + M.dot(E0);
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2, epsilon))
				return false;
		}
		// Seperating axis: E0 ª F1
		{
			const double p0 = 0;
			const double p1 = 0;
			const double p2 = -(N.dot(F1));
			const double q0 = E0.cross(F1).dot(D);
			const double q1 = q0 - M.dot(E0);
			const double q2 = q0;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2, epsilon))
				return false;
		}
		// Seperating axis: E0 ª F2
		{
			const double p0 = 0;
			const double p1 = 0;
			const double p2 = -(N.dot(F2));
			const double q0 = E0.cross(F2).dot(D);
			const double q1 = q0 - M.dot(E0);
			const double q2 = q1;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2, epsilon))
				return false;
		}
		// Seperating axis: E1 ª F0
		{
			const double p0 = 0;
			const double p1 = N.dot(F0);
			const double p2 = 0;
			const double q0 = E1.cross(F0).dot(D);
			const double q1 = q0;
			const double q2 = q0 + M.dot(E1);
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2, epsilon))
				return false;
		}
		// Seperating axis: E1 ª F1
		{
			const double p0 = 0;
			const double p1 = N.dot(F1);
			const double p2 = 0;
			const double q0 = E1.cross(F1).dot(D);
			const double q1 = q0 - M.dot(E1);
			const double q2 = q0;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2, epsilon))
				return false;
		}
		// Seperating axis: E1 ª F2
		{
			const double p0 = 0;
			const double p1 = N.dot(F2);
			const double p2 = 0;
			const double q0 = E1.cross(F2).dot(D);
			const double q1 = q0 - M.dot(E1);
			const double q2 = q1;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2, epsilon))
				return false;
		}
		// Seperating axis: E2 ª F0
		{
			const double p0 = 0;
			const double p1 = N.dot(F0);
			const double p2 = p1;
			const double q0 = E2.cross(F0).dot(D);
			const double q1 = q0;
			const double q2 = q0 + M.dot(E2);
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2, epsilon))
				return false;
		}
		// Seperating axis: E2 ª F1
		{
			const double p0 = 0;
			const double p1 = N.dot(F1);
			const double p2 = p1;
			const double q0 = E2.cross(F1).dot(D);
			const double q1 = q0 - M.dot(E2);
			const double q2 = q0;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2, epsilon))
				return false;
		}
		// Seperating axis: E2 ª F2
		{
			const double p0 = 0;
			const double p1 = N.dot(F2);
			const double p2 = p1;
			const double q0 = E2.cross(F2).dot(D);
			const double q1 = q0 - M.dot(E2);
			const double q2 = q1;
			if (areProjectionsSeparated(p0, p1, p2, q0, q1, q2, epsilon))
				return false;
		}
		return true;
	}

	static std::vector<std::pair<int, int>> getFlippedFaces(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixXi& F,
		const std::vector<Eigen::Vector2d>& hinges_faceIndex,
		const double epsilon)
	{
		Eigen::MatrixX3d N;
		igl::per_face_normals(V, F, N);
		std::vector<std::pair<int, int>> result;
		for (auto& h : hinges_faceIndex) {
			const int f1 = h[0];
			const int f2 = h[1];
			Eigen::RowVector3d N1 = N.row(f1).normalized();
			Eigen::RowVector3d N2 = N.row(f2).normalized();
			const double diff = (N1 + N2).squaredNorm();
			if (diff < epsilon) {
				result.push_back(std::pair<int, int>(f1, f2));
				//std::cout << "Found a flipped-face between: " << f1 << ", " << f2 << std::endl;
			}
		}
		return result;
	}
}

#define INPUT_MODEL_SCREEN -1
#define NOT_FOUND -1
#define ADD false
#define DELETE true

class UI {
public:
	app_utils::UserInterfaceOptions status;
	bool isActive;
	int Vertex_Index, Output_Index, Face_index;
	int down_mouse_x, down_mouse_y;
	bool ADD_DELETE;
	Eigen::Vector3f intersec_point;
	Eigen::Vector3f colorP, colorM, colorTry;
	std::vector<int> DFS_vertices_list;
	int DFS_Vertex_Index_FROM;

	UI() {
		status = app_utils::UserInterfaceOptions::NONE;
		isActive = false;
		Output_Index = Face_index = Vertex_Index = NOT_FOUND;
		down_mouse_x = down_mouse_y = NOT_FOUND;
		colorP = Eigen::Vector3f(51 / 255.0f, 1, 1);
		colorM = Eigen::Vector3f(1, 10 / 255.0f, 1);
		colorTry = Eigen::Vector3f(1, 200 / 255.0f, 1);
	}

	void updateVerticesListOfDFS(const Eigen::MatrixXi& F, const int numV, const int v_to) {
		Eigen::SparseMatrix<int> A;
		igl::adjacency_matrix(F, A);
		//convert sparse matrix into vector representation
		std::vector<std::vector<int>> adj;
		adj.resize(numV);
		for (int k = 0; k < A.outerSize(); ++k)
			for (Eigen::SparseMatrix<int>::InnerIterator it(A, k); it; ++it)
				adj[it.row()].push_back(it.col());
		//get the vertices list by DFS
		DFS_vertices_list = app_utils::findPathVertices_usingDFS(adj, DFS_Vertex_Index_FROM, v_to, numV);
	}

	bool isChoosingCluster() {
		return (status == app_utils::UserInterfaceOptions::ADJ_WEIGHTS && isActive && Face_index != NOT_FOUND);
	}
	bool isUsingDFS() {
		return status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR && ADD_DELETE == ADD && isActive;
	}
	bool isTranslatingVertex() {
		return (status == app_utils::UserInterfaceOptions::FIX_VERTICES && isActive);
	}
	bool isBrushingWeightInc() {
		return (status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_INCR && isActive);
	}
	bool isBrushingWeightDec() {
		return status == app_utils::UserInterfaceOptions::BRUSH_WEIGHTS_DECR && ADD_DELETE == DELETE && isActive;
	}
	bool isBrushing() {
		return (isBrushingWeightInc() || isBrushingWeightDec()) && Face_index != NOT_FOUND;
	}

	Eigen::RowVector3d getBrushColor(const Eigen::Vector3f& model_color) {
		if (ADD_DELETE == ADD && isBrushingWeightInc())
			return colorP.cast<double>().transpose();
		return model_color.cast<double>().transpose();
	}
	void clear() {
		DFS_vertices_list.clear();
		isActive = false;
		Output_Index = Face_index = Vertex_Index = NOT_FOUND;
		down_mouse_x = down_mouse_y = NOT_FOUND;
		DFS_Vertex_Index_FROM = NOT_FOUND;
	}
	
};

