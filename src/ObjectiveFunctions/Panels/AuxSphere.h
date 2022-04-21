#pragma once
#include "ObjectiveFunctions/Panels/AuxBasic.h"

namespace ObjectiveFunctions {
	namespace Panels {
		class AuxSphere: public AuxBasic
		{
		public:
			double w1 = 1, w2 = 100;
			AuxSphere(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F, const Cuda::PenaltyFunction type);
			~AuxSphere();
			virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
			virtual void gradient(Cuda::Array<double>& X, const bool update) override;
		};
	};
};

