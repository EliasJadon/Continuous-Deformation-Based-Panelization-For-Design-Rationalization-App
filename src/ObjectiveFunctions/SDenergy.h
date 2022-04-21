#pragma once
#include "ObjectiveFunction.h"

class SDenergy : public ObjectiveFunction {
public:
	Cuda::Array<double_3> D1d, D2d;
	Eigen::VectorXd restShapeArea;

	SDenergy(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~SDenergy();
	
	virtual double value(Cuda::Array<double>& curr_x, const bool update);
	virtual void gradient(Cuda::Array<double>& X, const bool update);
};