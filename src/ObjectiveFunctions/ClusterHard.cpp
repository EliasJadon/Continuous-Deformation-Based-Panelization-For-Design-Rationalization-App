#include "ClusterHard.h"

ClusterHard::ClusterHard(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F)
	: ObjectiveFunction{ V,F }
{
	name = "Cluster Hard";
	w = 0;
	
	std::cout << "\t" << name << " constructor" << std::endl;
}

ClusterHard::~ClusterHard() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void ClusterHard::updateClustering(const std::vector<std::vector<int>>& c) {
	m_value.lock();
	m_gradient.lock();
	clustering_faces_indices = c;
	m_gradient.unlock();
	m_value.unlock();
}

double ClusterHard::value(Cuda::Array<double>& curr_x, const bool update) {
	double value = 0;

	m_value.lock();
	for (std::vector<int>& cluster : clustering_faces_indices) {
		for (int f1 : cluster) {
			for (int f2 : cluster) {
				double_3 N1 = getN(curr_x, f1);
				double_3 N2 = getN(curr_x, f2);
				value += squared_norm(sub(N1, N2));
			}
		}
	}
	m_value.unlock();

	if (update)
		energy_value = value;
	return value;
}

void ClusterHard::gradient(Cuda::Array<double>& input_X, const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;
	
	m_gradient.lock();
	for (std::vector<int>& cluster : clustering_faces_indices) {
		for (int f1 : cluster) {
			for (int f2 : cluster) {
				double_3 N1 = getN(input_X, f1);
				double_3 N2 = getN(input_X, f2);
				double_3 diff = sub(N1, N2);
				grad.host_arr[f1 + mesh_indices.startNx] += 2 * diff.x;
				grad.host_arr[f1 + mesh_indices.startNy] += 2 * diff.y;
				grad.host_arr[f1 + mesh_indices.startNz] += 2 * diff.z;
				grad.host_arr[f2 + mesh_indices.startNx] += -2 * diff.x;
				grad.host_arr[f2 + mesh_indices.startNy] += -2 * diff.y;
				grad.host_arr[f2 + mesh_indices.startNz] += -2 * diff.z;
			}
		}
	}
	m_gradient.unlock();

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}
