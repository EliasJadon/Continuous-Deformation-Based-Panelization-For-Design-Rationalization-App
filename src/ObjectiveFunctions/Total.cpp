#include "ObjectiveFunctions/Total.h"

using namespace ObjectiveFunctions;

Total::Total(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) : ObjectiveFunctions::Basic{ V,F }
{
	name = "Total";
	objectiveList.clear();

	std::cout << Utils::ConsoleColor::yellow << "-------Energies, begin-------" << std::endl;
	aux_cylinder1		= std::make_shared<Panels::AuxCylinder1>(V, F, Cuda::PenaltyFunction::SIGMOID);
	aux_cylinder2		= std::make_shared<Panels::AuxCylinder2>(V, F, Cuda::PenaltyFunction::SIGMOID);
	aux_cylinder3		= std::make_shared<Panels::AuxCylinder3>(V, F, Cuda::PenaltyFunction::SIGMOID);
	aux_sphere			= std::make_shared<Panels::AuxSphere>(V, F, Cuda::PenaltyFunction::SIGMOID);
	aux_planar			= std::make_shared<Panels::AuxPlanar>(V, F, Cuda::PenaltyFunction::SIGMOID);
	planar				= std::make_shared<Panels::Planar>(V, F, Cuda::PenaltyFunction::SIGMOID);
	pin_chosen_vertices = std::make_shared<Deformation::PinChosenVertices>(V, F);
	pin_vertices		= std::make_shared<Deformation::PinVertices>(V, F);
	stvk				= std::make_shared<Deformation::STVK>(V, F);
	symmetric_dirichlet = std::make_shared<Deformation::SymmetricDirichlet>(V, F);
	uniform_smoothness	= std::make_shared<Deformation::UniformSmoothness>(V, F);
	round_radiuses		= std::make_shared<Fabrication::RoundRadiuses>(V, F);

	objectiveList.push_back(aux_cylinder1);
	objectiveList.push_back(aux_cylinder2);
	objectiveList.push_back(aux_cylinder3);
	objectiveList.push_back(aux_sphere);
	objectiveList.push_back(aux_planar);
	objectiveList.push_back(planar);
	objectiveList.push_back(stvk);
	objectiveList.push_back(symmetric_dirichlet);
	objectiveList.push_back(pin_vertices);
	objectiveList.push_back(pin_chosen_vertices);
	objectiveList.push_back(round_radiuses);
	objectiveList.push_back(uniform_smoothness);
	std::cout << "-------Energies, end-------" << Utils::ConsoleColor::white << std::endl;

	std::cout << "\t" << name << " constructor" << std::endl;
}

Total::~Total()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

double Total::value(Cuda::Array<double>& curr_x, const bool update)
{
	double value = 0;
	for (auto& obj : objectiveList)
		if (obj->w != 0)
			value += obj->w * obj->value(curr_x, update);
	
	if (update)
		energy_value = value;
	return value;
}

void Total::gradient(Cuda::Array<double>& X, const bool update)
{
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;

	for (auto& obj : objectiveList) {
		if (obj->w != 0) {
			obj->gradient(X, update);
			for (int i = 0; i < grad.size; i++)
				grad.host_arr[i] += obj->w * obj->grad.host_arr[i];
		}
	}
		
	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}