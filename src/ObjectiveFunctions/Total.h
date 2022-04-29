#pragma once
#include "ObjectiveFunctions/Basic.h"
#include "ObjectiveFunctions/Panels/AuxCylinder1.h"
#include "ObjectiveFunctions/Panels/AuxCylinder2.h"
#include "ObjectiveFunctions/Panels/AuxCylinder3.h"
#include "ObjectiveFunctions/Panels/AuxPlanar.h"
#include "ObjectiveFunctions/Panels/AuxSphere.h"
#include "ObjectiveFunctions/Panels/Planar.h"
#include "ObjectiveFunctions/Deformation/PinChosenVertices.h"
#include "ObjectiveFunctions/Deformation/PinVertices.h"
#include "ObjectiveFunctions/Deformation/STVK.h"
#include "ObjectiveFunctions/Deformation/SymmetricDirichlet.h"
#include "ObjectiveFunctions/Deformation/UniformSmoothness.h"
#include "ObjectiveFunctions/Fabrication/RoundRadiuses.h"

namespace ObjectiveFunctions {
	class Total : public ObjectiveFunctions::Basic
	{
	public:
		std::vector<std::shared_ptr<ObjectiveFunctions::Basic>> objectiveList;
		Total(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
		~Total();
		virtual double value(Cuda::Array<double>& curr_x, const bool update);
		virtual void gradient(Cuda::Array<double>& X, const bool update);

		std::shared_ptr <Panels::AuxSphere> aux_sphere;
		std::shared_ptr <Panels::AuxCylinder1> aux_cylinder1;
		std::shared_ptr <Panels::AuxCylinder2> aux_cylinder2;
		std::shared_ptr <Panels::AuxCylinder3> aux_cylinder3;
		std::shared_ptr <Panels::AuxPlanar> aux_planar;
		std::shared_ptr <Panels::Planar> planar;
		std::shared_ptr <Deformation::PinChosenVertices> pin_chosen_vertices;
		std::shared_ptr <Deformation::PinVertices> pin_vertices;
		std::shared_ptr <Deformation::STVK> stvk;
		std::shared_ptr <Deformation::SymmetricDirichlet> symmetric_dirichlet;
		std::shared_ptr <Deformation::UniformSmoothness> uniform_smoothness;
		std::shared_ptr <Fabrication::RoundRadiuses> round_radiuses;
	};
};