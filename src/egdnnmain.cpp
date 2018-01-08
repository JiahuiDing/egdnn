#include "egdnnmain.h"
#include "egdnn.h"
#include "test.h"
#include "helper.h"
using namespace EGDNN;

int main(int argc, char *argv[])
{
	int maxIter = 100;
	int batchSize = 100;
	int evolutionTime = 5;
	int populationSize = 3;
	
	int training_N;
	int test_N;
	int input_N;
	int output_N;
	
	std::vector<std::vector<double>> trainingSet;
	std::vector<std::vector<double>> trainingLabels;
	std::vector<std::vector<double>> testSet;
	std::vector<std::vector<double>> testLabels;
	
	read_mnist(trainingSet, trainingLabels, training_N, 
				testSet, testLabels, test_N, 
				input_N, output_N);
	
	/*
	SimpleNeuralNetwork(trainingSet, trainingLabels, training_N, 
						testSet, testLabels, test_N, 
						input_N, output_N, maxIter, batchSize);
	*/
	
	EvolutionaryGradientDescentNeuralNetwork(trainingSet, trainingLabels, training_N, 
											testSet, testLabels, test_N, 
											input_N, output_N, maxIter, batchSize, evolutionTime, populationSize);
	
	return 0;
}
