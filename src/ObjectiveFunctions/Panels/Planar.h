#pragma once
#include "ObjectiveFunctions/Panels/AuxBasic.h"

namespace ObjectiveFunctions {
	namespace Panels {
		class Planar : public ObjectiveFunctions::Panels::AuxBasic
		{
		private:
			Eigen::MatrixX3d normals, CurrV;
			Eigen::Matrix<double, 3, 9> dN_dx_perface(int hi);
			Eigen::Matrix< double, 6, 1> dm_dN(int hi);
			Eigen::Matrix<double, 6, 12> dN_dx_perhinge(int hi);

			int x_GlobInd(int index, int hi);
		public:
			Planar(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F, const Cuda::PenaltyFunction type);
			~Planar();
			virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
			virtual void gradient(Cuda::Array<double>& X, const bool update) override;
		};
	};
};