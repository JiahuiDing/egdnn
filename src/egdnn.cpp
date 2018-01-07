#include "egdnn.h"

namespace EGDNN
{
	void EvolutionaryGradientDescentNeuralNetwork(std::vector<std::vector<double>> trainingSet, std::vector<std::vector<double>> trainingLabels, int training_N, 
													std::vector<std::vector<double>> testSet, std::vector<std::vector<double>> testLabels, int test_N, 
													int input_N, int output_N, int maxIter, int batchSize, int populationSize)
	{
		srand(getpid());
		
		Network network;
		Neuron *input_neurons[input_N];
		Neuron *output_neurons[output_N];
		
		for(int i = 0; i < input_N; i++)
		{
			input_neurons[i] = new Neuron(-1, Neuron::input);
			network.AddNeuron(input_neurons[i]);
		}
		for(int i = 0; i < output_N; i++)
		{
			output_neurons[i] = new Neuron(i, Neuron::output);
			network.AddNeuron(output_neurons[i]);
		}
		
		/*
		// training & evolution
		struct timeval start, end;
		gettimeofday(&start, NULL);
		for(int iterCnt = 0; iterCnt < maxIter; iterCnt++)
		{
			int zeroCnt = 0;
			double error = 0;
			int rightCnt = 0;
		}
		*/
		
		
		// test
		double error = 0;
		int rightCnt = 0;
		for(int data_i = 0; data_i < test_N; data_i++)
		{
			for(int i = 0; i < input_N; i++)
			{
				input_neurons[i]->value = testSet[data_i][i];
			}
			for(int i = 0; i < output_N; i++)
			{
				output_neurons[i]->trueValue = testLabels[data_i][i];
			}
		
			network.ForwardPropagation();
		
			error += network.CalError();
			if(testLabels[data_i][network.CalMaxLabel()] > 0.5)
			{
				rightCnt++;
			}
		}
		std::cout << "test error : " << error / test_N << "\n";
		std::cout << "test accuracy : " << (double)rightCnt / test_N << "\n\n";
	}
}
