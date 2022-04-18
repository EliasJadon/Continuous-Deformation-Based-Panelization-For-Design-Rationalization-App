#pragma once
#include "ObjectiveFunction.h"

class fixRadius : public ObjectiveFunction {
public:
	int min = 2, max = 10;
	float alpha = 23;

	fixRadius(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~fixRadius();

	virtual double value(Cuda::Array<double>& curr_x, const bool update);
	virtual void gradient(Cuda::Array<double>& X, const bool update);
};