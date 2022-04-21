#include "ObjectiveFunctions/Panels/AuxBasic.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>

using namespace ObjectiveFunctions::Panels;

AuxBasic::AuxBasic(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixX3i& F,
	const Cuda::PenaltyFunction penaltyFunction) : ObjectiveFunction{ V,F }
{
	w = 1;
	name = "Aux Variables";

	//Initialize rest variables (X0) m
	calculateHinges();
	restAreaPerHinge.resize(num_hinges);
	igl::doublearea(restShapeV, restShapeF, restAreaPerFace);
	restAreaPerFace /= 2;
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		restAreaPerHinge(hi) = (restAreaPerFace(f0) + restAreaPerFace(f1)) / 2;
	}

	//Init Cuda variables
	const unsigned int numF = restShapeF.rows();
	const unsigned int numV = restShapeV.rows();
	const unsigned int numH = num_hinges;
	this->penaltyFunction = penaltyFunction;
	SigmoidParameter = 1;

	Cuda::AllocateMemory(weight_PerHinge, numH);
	Cuda::AllocateMemory(Sigmoid_PerHinge, numH);
	//init host buffers...
	for (int h = 0; h < num_hinges; h++) {
		weight_PerHinge.host_arr[h] = 1;
		Sigmoid_PerHinge.host_arr[h] = get_SigmoidParameter();
	}
	std::cout << "\t" << name << " constructor" << std::endl;
}

AuxBasic::~AuxBasic() {
	Cuda::FreeMemory(weight_PerHinge);
	Cuda::FreeMemory(Sigmoid_PerHinge);
	std::cout << "\t" << name << " destructor" << std::endl;
}

void AuxBasic::calculateHinges() {
	std::vector<std::vector<std::vector<int>>> TT;
	igl::triangle_triangle_adjacency(restShapeF, TT);
	assert(TT.size() == restShapeF.rows());
	hinges_faceIndex.clear();

	///////////////////////////////////////////////////////////
	//Part 1 - Find unique hinges
	for (int fi = 0; fi < TT.size(); fi++) {
		std::vector< std::vector<int>> CurrFace = TT[fi];
		assert(CurrFace.size() == 3 && "Each face should be a triangle (not square for example)!");
		for (std::vector<int> hinge : CurrFace) {
			if (hinge.size() == 1) {
				//add this "hinge"
				int FaceIndex1 = fi;
				int FaceIndex2 = hinge[0];

				if (FaceIndex2 < FaceIndex1) {
					//Skip
					//This hinge already exists!
					//Empty on purpose
				}
				else {
					hinges_faceIndex.push_back(Eigen::Vector2d(FaceIndex1, FaceIndex2));
				}
			}
			else if (hinge.size() == 0) {
				//Skip
				//This triangle has no another adjacent triangle on that edge
				//Empty on purpose
			}
			else {
				//We shouldn't get here!
				//The mesh is invalid
				assert("Each triangle should have only one adjacent triangle on each edge!");
			}

		}
	}
	num_hinges = hinges_faceIndex.size();

	///////////////////////////////////////////////////////////
	//Part 2 - Find x0,x1,x2,x3 indecis for each hinge
	x0_GlobInd.resize(num_hinges);
	x1_GlobInd.resize(num_hinges);
	x2_GlobInd.resize(num_hinges);
	x3_GlobInd.resize(num_hinges);
	x0_LocInd.resize(num_hinges, 2); x0_LocInd.setConstant(-1);
	x1_LocInd.resize(num_hinges, 2); x1_LocInd.setConstant(-1);
	x2_LocInd.resize(num_hinges, 2); x2_LocInd.setConstant(-1);
	x3_LocInd.resize(num_hinges, 2); x3_LocInd.setConstant(-1);

	for (int hi = 0; hi < num_hinges; hi++) {
		//first triangle vertices
		int v1 = restShapeF(hinges_faceIndex[hi](0), 0);
		int v2 = restShapeF(hinges_faceIndex[hi](0), 1);
		int v3 = restShapeF(hinges_faceIndex[hi](0), 2);
		//second triangle vertices
		int V1 = restShapeF(hinges_faceIndex[hi](1), 0);
		int V2 = restShapeF(hinges_faceIndex[hi](1), 1);
		int V3 = restShapeF(hinges_faceIndex[hi](1), 2);

		/*
		* Here we should find x0,x1,x2,x3
		* from the given two triangles (v1,v2,v3),(V1,V2,V3)
		*
		*	x0--x2
		*  / \ /
		* x3--x1
		*
		*/
		if (v1 != V1 && v1 != V2 && v1 != V3) {
			x2_GlobInd(hi) = v1; x2_LocInd(hi, 0) = 0;
			x0_GlobInd(hi) = v2; x0_LocInd(hi, 0) = 1;
			x1_GlobInd(hi) = v3; x1_LocInd(hi, 0) = 2;

			if (V1 != v1 && V1 != v2 && V1 != v3) {
				x3_GlobInd(hi) = V1; x3_LocInd(hi, 1) = 0;
			}
			else if (V2 != v1 && V2 != v2 && V2 != v3) {
				x3_GlobInd(hi) = V2; x3_LocInd(hi, 1) = 1;
			}
			else {
				x3_GlobInd(hi) = V3; x3_LocInd(hi, 1) = 2;
			}
		}
		else if (v2 != V1 && v2 != V2 && v2 != V3) {
			x2_GlobInd(hi) = v2; x2_LocInd(hi, 0) = 1;
			x0_GlobInd(hi) = v1; x0_LocInd(hi, 0) = 0;
			x1_GlobInd(hi) = v3; x1_LocInd(hi, 0) = 2;

			if (V1 != v1 && V1 != v2 && V1 != v3) {
				x3_GlobInd(hi) = V1; x3_LocInd(hi, 1) = 0;
			}	
			else if (V2 != v1 && V2 != v2 && V2 != v3) {
				x3_GlobInd(hi) = V2; x3_LocInd(hi, 1) = 1;
			}
			else {
				x3_GlobInd(hi) = V3; x3_LocInd(hi, 1) = 2;
			}
		}
		else {
			x2_GlobInd(hi) = v3; x2_LocInd(hi, 0) = 2;
			x0_GlobInd(hi) = v1; x0_LocInd(hi, 0) = 0;
			x1_GlobInd(hi) = v2; x1_LocInd(hi, 0) = 1;

			if (V1 != v1 && V1 != v2 && V1 != v3) {
				x3_GlobInd(hi) = V1; x3_LocInd(hi, 1) = 0;
			}
			else if (V2 != v1 && V2 != v2 && V2 != v3) {
				x3_GlobInd(hi) = V2; x3_LocInd(hi, 1) = 1;
			}
			else {
				x3_GlobInd(hi) = V3; x3_LocInd(hi, 1) = 2;
			}
		}

		if (V1 == x0_GlobInd(hi))
			x0_LocInd(hi, 1) = 0;
		else if (V2 == x0_GlobInd(hi))
			x0_LocInd(hi, 1) = 1;
		else if (V3 == x0_GlobInd(hi))
			x0_LocInd(hi, 1) = 2;

		if (V1 == x1_GlobInd(hi))
			x1_LocInd(hi, 1) = 0;
		else if (V2 == x1_GlobInd(hi))
			x1_LocInd(hi, 1) = 1;
		else if (V3 == x1_GlobInd(hi))
			x1_LocInd(hi, 1) = 2;
	}
}

void AuxBasic::Incr_HingesWeights(const std::vector<int> faces_indices,const double add)
{
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H) {
			if (weight_PerHinge.host_arr[hi] != 0) {
				weight_PerHinge.host_arr[hi] += add;
				if (weight_PerHinge.host_arr[hi] <= 1) {
					weight_PerHinge.host_arr[hi] = 1;
				}
				Sigmoid_PerHinge.host_arr[hi] = 1;
			}
		}
	}
}

void AuxBasic::setZero_HingesWeights(const std::vector<int> vertices_indices) {
	for (int vi : vertices_indices) {
		int hi = OptimizationUtils::VertexToHinge_indices(x0_GlobInd, x1_GlobInd, vertices_indices, vi);
		if (hi >= 0 && hi < num_hinges) {
			weight_PerHinge.host_arr[hi] = 0;
		}
	}
}

void AuxBasic::setOne_HingesWeights(const std::vector<int> faces_indices)
{
	for (int fi : faces_indices) {
		std::vector<int> H = OptimizationUtils::FaceToHinge_indices(hinges_faceIndex, faces_indices, fi);
		for (int hi : H)
			if (weight_PerHinge.host_arr[hi] == 0)
				weight_PerHinge.host_arr[hi] = 1;
	}
}

void AuxBasic::Clear_HingesWeights() {
	for (int hi = 0; hi < num_hinges; hi++) {
		weight_PerHinge.host_arr[hi] = 1;
	}
}

void AuxBasic::Inc_SigmoidParameter() {
	SigmoidParameter *= 2;
	for (int hi = 0; hi < mesh_indices.num_hinges; hi++) {
		Sigmoid_PerHinge.host_arr[hi] *= 2;
	}
}
void AuxBasic::Dec_SigmoidParameter(const double target) {
	if (SigmoidParameter > target)
		SigmoidParameter /= 2;
	for (int hi = 0; hi < mesh_indices.num_hinges; hi++) {
		if (Sigmoid_PerHinge.host_arr[hi] > target)
			Sigmoid_PerHinge.host_arr[hi] /= 2;
	}
}
double AuxBasic::get_SigmoidParameter() {
	return SigmoidParameter;
}

double AuxBasic::Phi(
	const double x,
	const double SigmoidParameter,
	const Cuda::PenaltyFunction penaltyFunction)
{
	if (penaltyFunction == Cuda::PenaltyFunction::SIGMOID) {
		double x2 = pow(x, 2);
		return x2 / (x2 + SigmoidParameter);
	}
	else if (penaltyFunction == Cuda::PenaltyFunction::QUADRATIC)
		return pow(x, 2);
	else if (penaltyFunction == Cuda::PenaltyFunction::EXPONENTIAL)
		return exp(x * x);
	else {
		exit(EXIT_FAILURE);
	}
}

double AuxBasic::dPhi_dm(
	const double x,
	const double SigmoidParameter,
	const Cuda::PenaltyFunction penaltyFunction)
{
	if (penaltyFunction == Cuda::PenaltyFunction::SIGMOID)
		return (2 * x * SigmoidParameter) / pow(x * x + SigmoidParameter, 2);
	if (penaltyFunction == Cuda::PenaltyFunction::QUADRATIC)
		return 2 * x;
	if (penaltyFunction == Cuda::PenaltyFunction::EXPONENTIAL)
		return 2 * x * exp(x * x);
}