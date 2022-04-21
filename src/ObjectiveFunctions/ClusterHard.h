#pragma once
#include "ObjectiveFunction.h"
#include <mutex>

class ClusterHard : public ObjectiveFunction {
private:
	std::mutex m_value, m_gradient;
	std::vector<std::vector<int>> clustering_faces_indices;
public:
	ClusterHard(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~ClusterHard();

	void updateClustering(const std::vector<std::vector<int>>& c);

	virtual double value(Cuda::Array<double>& curr_x, const bool update);
	virtual void gradient(Cuda::Array<double>& X, const bool update);
};