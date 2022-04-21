#pragma once
#include "ObjectiveFunctions/Panels/AuxBasic.h"

class AuxBendingNormal : public ObjectiveFunctions::Panels::AuxBasic
{	
public:
	double w1 = 1, w2 = 100, w3 = 100;
	AuxBendingNormal(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F, const Cuda::PenaltyFunction type);
	~AuxBendingNormal();
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual void gradient(Cuda::Array<double>& X, const bool update) override;
};

