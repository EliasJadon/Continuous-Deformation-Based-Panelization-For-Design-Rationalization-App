#pragma once
#include "AuxVariables.h"

class BendingNormal : public AuxVariables
{
private:
	Eigen::MatrixX3d normals, CurrV;
	Eigen::Matrix<double, 3, 9> dN_dx_perface(int hi);
	Eigen::Matrix< double, 6, 1> dm_dN(int hi);
	Eigen::Matrix<double, 6, 12> dN_dx_perhinge(int hi);

	int x_GlobInd(int index, int hi);
public:
	BendingNormal(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F, const Cuda::PenaltyFunction type);
	~BendingNormal();
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual void gradient(Cuda::Array<double>& X, const bool update) override;
};