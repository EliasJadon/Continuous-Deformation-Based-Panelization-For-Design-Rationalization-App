#pragma once
#include <limits>

static void RadiusClustering(
	const Eigen::MatrixX4d& values,
	const double MSE_threshold,
	std::vector<std::vector<int>>& final_clusters)
{
	std::vector<Eigen::RowVector4d> values_representors; values_representors.clear();
	final_clusters.clear();
	for (int fi = 0; fi < values.rows(); fi++) {
		int argmin = -1;
		double min = std::numeric_limits<double>::max();
		for (int i = 0; i < values_representors.size(); i++) {
			double MSE = (values_representors[i] - values.row(fi)).squaredNorm();
			if ((MSE < MSE_threshold) && (MSE < min)) {
				argmin = i;
				min = MSE;
			}
		}
		if (argmin != -1) { final_clusters[argmin].push_back(fi); }
		else {
			values_representors.push_back(values.row(fi));
			final_clusters.push_back({ fi });
		}
	}
}


class DoubleLinkedList {
public:
	DoubleLinkedList* next;
	DoubleLinkedList* prev;
	Eigen::RowVector4d Avg, Sum;
	unsigned int count;
	std::vector<int> faces_list;

	DoubleLinkedList(const Eigen::RowVector4d value, const int fi) {
		Avg = Sum = value;
		count = 1;
		faces_list = { fi };
		next = prev = NULL;
	}
	void print() {
		std::cout << "-------------------------\n";
		std::cout << "avg = " << Avg << std::endl;
		std::cout << "Sum = " << Sum << std::endl;
		std::cout << "count = " << count << std::endl;
		std::cout << "faces_list = ";
		for (auto fi : faces_list)
			std::cout << fi << " ";
		std::cout << std::endl;
	}
};

class Agglomerative_hierarchical_clustering {
private:
	DoubleLinkedList* head, *tail;

	void print() {
		DoubleLinkedList* p1 = head;
		std::cout << "===============================================\n";
		while (p1 != NULL) {
			p1->print();
			p1 = p1->next;
		}
	}

	bool findMinimum(const double MinDistanceAllowed) {
		DoubleLinkedList* p1, * p2;
		DoubleLinkedList* argmin1 = NULL;
		DoubleLinkedList* argmin2 = NULL;
		double min_value = std::numeric_limits<double>::max();
		p1 = head;
		while (p1 != NULL) {
			p2 = p1->next;
			while (p2 != NULL) {
				double currDistance = (p1->Avg - p2->Avg).squaredNorm();
				if ((currDistance < MinDistanceAllowed) && (currDistance < min_value)) {
					argmin1 = p1;
					argmin2 = p2;
					min_value = currDistance;
				}
				p2 = p2->next;
			}
			p1 = p1->next;
		}
		//if another minimum found, merge them...
		if (argmin1 != NULL && argmin2 != NULL) {
			merge(argmin1, argmin2);
			return true;
		}
		//else finish the process
		return false;
	}

	void addNode(const Eigen::RowVector4d value, const int fi) {
		DoubleLinkedList* pointer = new DoubleLinkedList(value, fi);
		if (head == NULL || tail == NULL) {
			head = tail = pointer;
			pointer->next = NULL;
			pointer->prev = NULL;
			return;
		}
		pointer->next = head;
		pointer->prev = NULL;
		head->prev = pointer;
		head = pointer;
		return;
	}

	void merge(DoubleLinkedList* A, DoubleLinkedList* B) {
		A->Sum += B->Sum;
		A->count += B->count;
		A->Avg = A->Sum / A->count;
		for (int fi : B->faces_list) {
			A->faces_list.push_back(fi);
		}
		removeNode(B);
	}

	void removeNode(DoubleLinkedList* node) {
		if (node == NULL) {
			head = tail = NULL;
			return;
		}
		else if (node->next == NULL && node->prev == NULL) {
			delete node;
			head = tail = NULL;
		}
		else if (node->next == NULL && node->prev != NULL) {
			tail = node->prev;
			node->prev->next = NULL;
			delete node;
		}
		else if (node->next != NULL && node->prev == NULL) {
			head = node->next;
			node->next->prev = NULL;
			delete node;
		}
		else {
			node->next->prev = node->prev;
			node->prev->next = node->next;
			delete node;
		}
	}

public:
	Agglomerative_hierarchical_clustering(
		const Eigen::MatrixX4d& values, 
		const double MinDistanceAllowed, 
		const int num_faces,
		std::vector<std::vector<int>>& final_clusters)
	{
		//init
		head = tail = NULL;
		for (int fi = 0; fi < values.rows(); fi++) {
			addNode(values.row(fi), fi);
		}
		//Run
		int count = 0;
		while (findMinimum(MinDistanceAllowed)) {
			std::cout << count++ << "\n";
		}
		//Get the clusters
		final_clusters.clear();
		DoubleLinkedList* p1 = head;
		while (p1 != NULL) {
			final_clusters.push_back(p1->faces_list);
			p1 = p1->next;
		}
	}
	~Agglomerative_hierarchical_clustering() {
		while (head != NULL && tail != NULL)
			removeNode(head);
	}
};

class Clustering_Colors {
private:
	std::vector<Eigen::Vector3d> colors;
	//random number from 0 to 1
	double getRand() { return (double)rand() / RAND_MAX; }
public:
	void changeColors() {
		for (auto& c : colors)
			c = Eigen::Vector3d(getRand(), getRand(), getRand());
	}
	void getFacesColors(
		const std::vector<std::vector<int>>& final_clusters, 
		const int num_faces,
		const double w,
		Eigen::MatrixX3d& colors_per_face)
	{
		//prepare matrix of colors for each face
		colors_per_face.resize(num_faces, 3);
		colors_per_face.setZero();
		for (int ci = 0; ci < final_clusters.size(); ci++) {
			if (colors.size() <= ci)
				colors.push_back(Eigen::Vector3d(getRand(), getRand(), getRand()));
			for (int fi : final_clusters[ci])
				colors_per_face.row(fi) = colors[ci];
		}
		//Add Brightness according to user weight...
		for (int fi = 0; fi < num_faces; fi++)
			for (int col = 0; col < 3; col++)
				colors_per_face(fi, col) = (w * colors_per_face(fi, col)) + (1 - w);
	}
};
