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
	void EvolutionaryGradientDescentNeuralNetwork(std::vector<std::vector<double>> trainingSet, std::vector<std::vector<double>> trainingLabels, int training_N, 
													std::vector<std::vector<double>> testSet, std::vector<std::vector<double>> testLabels, int test_N, 
													int input_N, int output_N, int maxIter, int batchSize, int evolutionTime, int populationSize, double learning_rate, double velocity_decay);
}

#endif
