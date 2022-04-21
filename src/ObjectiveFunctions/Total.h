#pragma once
#include "ObjectiveFunctions/Basic.h"

class TotalObjective : public ObjectiveFunction
{
public:
	double value_print(Cuda::Array<double>& curr_x, const bool update);
	std::vector<std::shared_ptr<ObjectiveFunction>> objectiveList;
	TotalObjective(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~TotalObjective();
	virtual double value(Cuda::Array<double>& curr_x, const bool update);
	virtual void gradient(Cuda::Array<double>& X, const bool update);
};