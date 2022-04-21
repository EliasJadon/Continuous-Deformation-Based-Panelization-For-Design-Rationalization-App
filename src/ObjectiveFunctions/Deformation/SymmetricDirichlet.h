#pragma once
#include "ObjectiveFunctions/Basic.h"

namespace ObjectiveFunctions {
	namespace Deformation {
		class SymmetricDirichlet : public ObjectiveFunctions::Basic {
		public:
			Cuda::Array<double_3> D1d, D2d;
			Eigen::VectorXd restShapeArea;

			SymmetricDirichlet(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
			~SymmetricDirichlet();

			virtual double value(Cuda::Array<double>& curr_x, const bool update);
			virtual void gradient(Cuda::Array<double>& X, const bool update);
		};
	};
};