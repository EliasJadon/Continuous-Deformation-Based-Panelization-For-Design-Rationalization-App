#pragma once
#include "AuxVariables.h"

class AuxSpherePerHinge : public AuxVariables
{		
public:
	double w1 = 1, w2 = 100;
	AuxSpherePerHinge(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F, const Cuda::PenaltyFunction type);
	~AuxSpherePerHinge();
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual void gradient(Cuda::Array<double>& X, const bool update) override;
};

