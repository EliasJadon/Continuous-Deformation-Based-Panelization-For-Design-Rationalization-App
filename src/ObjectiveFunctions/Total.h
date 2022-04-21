#pragma once
#include "ObjectiveFunctions/Basic.h"

namespace ObjectiveFunctions {
	class Total : public ObjectiveFunctions::Basic
	{
	public:
		std::vector<std::shared_ptr<ObjectiveFunctions::Basic>> objectiveList;
		Total(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
		~Total();
		virtual double value(Cuda::Array<double>& curr_x, const bool update);
		virtual void gradient(Cuda::Array<double>& X, const bool update);
	};
};