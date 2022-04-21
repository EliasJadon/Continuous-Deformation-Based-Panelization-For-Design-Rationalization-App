#include "ObjectiveFunctions/Deformation/PinChosenVertices.h"

FixChosenConstraints::FixChosenConstraints(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F)
	: ObjectiveFunction{ V,F }
{
    name = "Fix Chosen Vertices";
	w = 500;
	Constraints_Position.resize(V.rows(), 3);
	std::cout << "\t" << name << " constructor" << std::endl;
}

FixChosenConstraints::~FixChosenConstraints() 
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void FixChosenConstraints::insertConstraint(const int new_vertex, const Eigen::MatrixX3d& V)
{
	m_value.lock();
	m_gradient.lock();
	Constraints_indices.insert(new_vertex);
	Constraints_Position.row(new_vertex) = V.row(new_vertex);
	m_gradient.unlock();
	m_value.unlock();
}

void FixChosenConstraints::translateConstraint(const int vertex, const Eigen::RowVector3d& translation)
{
	m_value.lock();
	m_gradient.lock();
	Constraints_Position.row(vertex) += translation;
	m_gradient.unlock();
	m_value.unlock();
}

void FixChosenConstraints::eraseConstraint(const int vertex)
{
	m_value.lock();
	m_gradient.lock();
	Constraints_indices.erase(vertex);
	m_gradient.unlock();
	m_value.unlock();
}

void FixChosenConstraints::clearConstraints()
{
	m_value.lock();
	m_gradient.lock();
	Constraints_indices.clear();
	m_gradient.unlock();
	m_value.unlock();
}

std::set<int> FixChosenConstraints::getConstraintsIndices() {
	return Constraints_indices;
}

double FixChosenConstraints::value(Cuda::Array<double>& curr_x, const bool update)
{
	m_value.lock();
	double value = 0;
	for (int v_index : Constraints_indices) {
		double_3 Vi = getV(curr_x, v_index);
		value += squared_norm(sub(Vi, Constraints_Position.row(v_index)));
	}
	m_value.unlock();

	if (update)
		energy_value = value;
	return value;
}

void FixChosenConstraints::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;

	m_gradient.lock();
	for (int v_index : Constraints_indices) {
		grad.host_arr[v_index + mesh_indices.startVx] = 2 * (X.host_arr[v_index + mesh_indices.startVx] - Constraints_Position(v_index, 0));
		grad.host_arr[v_index + mesh_indices.startVy] = 2 * (X.host_arr[v_index + mesh_indices.startVy] - Constraints_Position(v_index, 1));
		grad.host_arr[v_index + mesh_indices.startVz] = 2 * (X.host_arr[v_index + mesh_indices.startVz] - Constraints_Position(v_index, 2));
	}
	m_gradient.unlock();

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}