#include "egdnn.h"

namespace EGDNN
{
	void EvolutionaryGradientDescentNeuralNetwork(std::vector<std::vector<double>> trainingSet, std::vector<std::vector<double>> trainingLabels, int training_N, 
													std::vector<std::vector<double>> testSet, std::vector<std::vector<double>> testLabels, int test_N, 
													int input_N, int output_N, int maxIter, int batchSize, int populationSize)
	{
		srand(getpid());
		
		Network *network[populationSize];
		for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
		{
			network[networkCnt] = new Network();
			for(int i = 0; i < input_N; i++)
			{
				network[networkCnt]->AddInputNeuron(new Neuron(-1, Neuron::input));
			}
			for(int i = 0; i < output_N; i++)
			{
				network[networkCnt]->AddOutputNeuron(new Neuron(i, Neuron::output));
			}
			network[networkCnt]->Mutation();
		}
		
		/*
		// training
		struct timeval start, end;
		gettimeofday(&start, NULL);
		for(int iterCnt = 0; iterCnt < maxIter; iterCnt++)
		{
			double error = 0;
			int rightCnt = 0;
			for(int data_i = 0; data_i < training_N; data_i++)
			{
				network->SetInputValue(trainingSet[data_i]);
				network->SetOutputValue(trainingLabels[data_i]);
				
				network->ForwardPropagation();
				network->BackPropagation();
				
				error += network->CalError();
				if(trainingLabels[data_i][network->CalMaxLabel()] > 0.5)
				{
					rightCnt++;
				}
				
				if(data_i % batchSize == 0)
				{
				}
			}
		}
		
		
		// test
		double error = 0;
		int rightCnt = 0;
		for(int data_i = 0; data_i < test_N; data_i++)
		{
			network->SetInputValue(testSet[data_i]);
			network->SetOutputValue(testLabels[data_i]);
			
			network->ForwardPropagation();
			
			error += network->CalError();
			if(testLabels[data_i][network->CalMaxLabel()] > 0.5)
			{
				rightCnt++;
			}
		}
		std::cout << "test error : " << error / test_N << "\n";
		std::cout << "test accuracy : " << (double) rightCnt / test_N << "\n\n";
		*/
	}
}
