#include "egdnnmain.h"
#include "egdnn.h"
#include "test.h"
#include "helper.h"
using namespace EGDNN;

// target function
int func(int x[])
{
	return ( x[0] || x[1] ) && (x[2] || x[3]);
}

void read_func(std::vector<std::vector<double>> &trainingSet, std::vector<std::vector<double>> &trainingLabels, int &training_N, 
				std::vector<std::vector<double>> &testSet, std::vector<std::vector<double>> &testLabels, int &test_N, 
				int &input_N, int &output_N)
{
	std::cout << "Begin loading func dataset!\n";

	training_N = 16;
	test_N = training_N;
	input_N = 4;
	output_N = 2;
	
	trainingSet.resize(training_N);
	trainingLabels.resize(training_N);
	for(int i = 0; i < training_N; i++)
	{
		trainingSet[i].resize(input_N);
		trainingLabels[i].resize(output_N);
		
		int x[input_N];
		for(int j = 0; j < input_N; j++)
		{
			x[j] = (i >> j) & 1;
			trainingSet[i][j] = x[j];
		}
		
		int y = func(x);
		trainingLabels[i][0] = 0;
		trainingLabels[i][1] = 0;
		trainingLabels[i][y] = 1;
	}
	
	testSet.resize(test_N);
	testLabels.resize(test_N);
	for(int i = 0; i < test_N; i++)
	{
		testSet[i].resize(input_N);
		testLabels[i].resize(output_N);
		
		for(int j = 0; j < input_N; j++)
		{
			testSet[i][j] = trainingSet[i][j];
		}
		
		for(int j = 0; j < output_N; j++)
		{
			testLabels[i][j] = trainingLabels[i][j];
		}
	}
	
	std::cout << "Finish loading func dataset!\n";
	
}

int main(int argc, char *argv[])
{	
	int maxIter = 1000000;
	int batchSize = 100;
	int evolutionTime = 20;
	int populationSize = 2;
	double learning_rate = 1e-3;
	double velocity_decay = 0.9;
	double regularization_l2 = 0.5;
	double gradientClip = 1;
	
	int training_N;
	int test_N;
	int input_N;
	int output_N;
	
	std::vector<std::vector<double>> trainingSet;
	std::vector<std::vector<double>> trainingLabels;
	std::vector<std::vector<double>> testSet;
	std::vector<std::vector<double>> testLabels;
	
	/*
	read_func(trainingSet, trainingLabels, training_N, 
				testSet, testLabels, test_N, 
				input_N, output_N);
	*/
	
	read_mnist(trainingSet, trainingLabels, training_N, 
				testSet, testLabels, test_N, 
				input_N, output_N);
	
	
	/*
	SimpleNeuralNetwork(trainingSet, trainingLabels, training_N, 
						testSet, testLabels, test_N, 
						input_N, output_N, maxIter, batchSize, 
						learning_rate, velocity_decay, regularization_l2, gradientClip);
	*/
	
	/*
	std::cout << "training_N : " << training_N << "\n";
	std::cout << "test_N : " << test_N << "\n";
	std::cout << "input_N : " << input_N << "\n";
	std::cout << "output_N : " << output_N << "\n";
	
	for(int i = 0; i < training_N; i++)
	{
		for(int j = 0; j < input_N; j++)
		{
			std::cout << trainingSet[i][j] << " ";
		}
		
		std::cout << " :  ";
		
		for(int j = 0; j < output_N; j++)
		{
			std::cout << trainingLabels[i][j] << " ";
		}
		
		std::cout << "\n";
	}
	*/
	
	Egdnn * model = new Egdnn(input_N, output_N, populationSize, learning_rate, velocity_decay, regularization_l2, gradientClip);
	model->fit(trainingSet, trainingLabels, training_N, maxIter, batchSize, evolutionTime);
	model->test(testSet, testLabels, test_N);
	
	return 0;
}
