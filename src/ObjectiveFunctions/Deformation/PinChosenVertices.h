#pragma once
#include "ObjectiveFunctions/Basic.h"
#include <mutex>

namespace ObjectiveFunctions {
	namespace Deformation {
		class PinChosenVertices: public Basic
		{
		private:
			std::mutex m_value, m_gradient;
			std::set<int> Constraints_indices;
			Eigen::MatrixX3d Constraints_Position;
		public:
			PinChosenVertices(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
			~PinChosenVertices();

			virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
			virtual void gradient(Cuda::Array<double>& X, const bool update) override;
			void insertConstraint(const int new_vertex, const Eigen::MatrixX3d& V);
			void translateConstraint(const int vertex, const Eigen::RowVector3d& pos);
			void eraseConstraint(const int vertex_index);
			void clearConstraints();
			std::set<int> getConstraintsIndices();
		};
	};
};