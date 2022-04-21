#pragma once
#include "ObjectiveFunctions/Basic.h"

namespace ObjectiveFunctions {
	namespace Fabrication {
		class RoundRadiuses : public Basic {
		public:
			int min = 2, max = 10;
			float alpha = 23;

			RoundRadiuses(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
			~RoundRadiuses();

			virtual double value(Cuda::Array<double>& curr_x, const bool update);
			virtual void gradient(Cuda::Array<double>& X, const bool update);
		};
	};
};