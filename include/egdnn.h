#ifndef _EGDNN_H_
#define _EGDNN_H_

#include "neuron.h"
#include "connection.h"
#include "network.h"
#include "helper.h"
#include <iostream>
#include <cstdio>
#include <iomanip>
#include <unistd.h>
#include <cmath>
#include <vector>
#include <sys/time.h>

namespace EGDNN
{
	class Egdnn
	{
		public:
		int input_N;
		int output_N;
		int populationSize;
		
		double learning_rate;
		double velocity_decay;
		double regularization_l2;
		double gradientClip;
		
		std::vector<Network *> network;
		
		Egdnn(int input_N, int output_N, int populationSize, double learning_rate, double velocity_decay, double regularization_l2, double gradientClip);
		void fit(std::vector<std::vector<double>> trainingSet, std::vector<std::vector<double>> trainingLabels, int maxIter, int batchSize, int evolutionTime);
		void test(std::vector<std::vector<double>> testSet, std::vector<std::vector<double>> testLabels);
		std::vector<double> predict(std::vector<double> data);
	};
}

#endif
