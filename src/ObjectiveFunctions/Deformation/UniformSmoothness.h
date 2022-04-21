#pragma once
#include "ObjectiveFunctions/Basic.h"

namespace ObjectiveFunctions {
	namespace Deformation {
		class UniformSmoothness : public ObjectiveFunctions::Basic {
		public:
			UniformSmoothness(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
			~UniformSmoothness();
			Eigen::SparseMatrix<double> L, L2;

			virtual double value(Cuda::Array<double>& curr_x, const bool update);
			virtual void gradient(Cuda::Array<double>& X, const bool update);
		};
	};
};