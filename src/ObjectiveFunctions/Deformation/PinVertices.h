#pragma once
#include "ObjectiveFunctions/Basic.h"

namespace ObjectiveFunctions {
	namespace Deformation {
		class PinVertices : public ObjectiveFunctions::Basic
		{
		public:
			PinVertices(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
			~PinVertices();

			virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
			virtual void gradient(Cuda::Array<double>& X, const bool update) override;
		};
	};
};