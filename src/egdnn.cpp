#include "egdnn.h"

namespace EGDNN
{
	void EvolutionaryGradientDescentNeuralNetwork(std::vector<std::vector<double>> trainingSet, std::vector<std::vector<double>> trainingLabels, int training_N, 
													std::vector<std::vector<double>> testSet, std::vector<std::vector<double>> testLabels, int test_N, 
													int input_N, int output_N, int maxIter, int batchSize, int evolutionTime, int populationSize)
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
			network[networkCnt]->Mutate();
		}
		
		// training
		struct timeval start, end;
		gettimeofday(&start, NULL);
		for(int iterCnt = 0; iterCnt < maxIter; iterCnt++)
		{
			double error[populationSize];
			int rightCnt[populationSize];
			for(int i = 0; i < populationSize; i++)
			{
				error[i] = 0;
				rightCnt[i] = 0;
			}
			
			for(int evolutionCnt = 0; evolutionCnt < evolutionTime; evolutionCnt++)
			{
				for(int batchCnt = 0; batchCnt < batchSize; batchCnt++)
				{
					int data_i = rand() % training_N;
					for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
					{
						network[networkCnt]->SetInputValue(trainingSet[data_i]);
						network[networkCnt]->SetOutputValue(trainingLabels[data_i]);
						network[networkCnt]->ForwardPropagation();
						network[networkCnt]->BackPropagation();
						
						error[networkCnt] += network[networkCnt]->CalError();
						if(trainingLabels[data_i][network[networkCnt]->CalMaxLabel()] > 0.5)
						{
							rightCnt[networkCnt]++;
						}
					}
				}
				
				/*
				std::cout << "iter " << iterCnt << " evolution " << evolutionCnt << " performance : \n";
				for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
				{
					std::cout << "network " << networkCnt << " : ";
					std::cout << "error " << error[networkCnt] / ((evolutionCnt + 1) * batchSize) << " , ";
					std::cout << "accuracy " << (double)rightCnt[networkCnt] / ((evolutionCnt + 1) * batchSize) << "\n";
				}
				gettimeofday(&end, NULL);
				int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
				std::cout << "time : " << timeuse / 1000 << " ms\n\n";
				gettimeofday(&start, NULL);
				*/
				
				for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
				{
					network[networkCnt]->UpdateWeight();
				}
			}
			
			int maxRightCnt = -1;
			int bestNetwork = -1;
			std::cout << "iter " << iterCnt << " performance : \n";
			for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
			{
				std::cout << "network " << networkCnt << " : ";
				std::cout << "error " << error[networkCnt] / (evolutionTime * batchSize) << " , ";
				std::cout << "accuracy " << (double)rightCnt[networkCnt] / (evolutionTime * batchSize) << "\n";
				
				if(rightCnt[networkCnt] > maxRightCnt)
				{
					maxRightCnt = rightCnt[networkCnt];
					bestNetwork = networkCnt;
				}
			}
			gettimeofday(&end, NULL);
			int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
			std::cout << "time : " << timeuse / 1000 << " ms\n\n";
			gettimeofday(&start, NULL);
			
			// kill all networks except the best
			for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
			{
				if(networkCnt != bestNetwork)
				{
					delete network[networkCnt];
				}
			}
			
			// reproduce
			network[0] = network[bestNetwork];
			for(int networkCnt = 1; networkCnt < populationSize; networkCnt++)
			{
				network[networkCnt] = network[0]->copy();
				network[networkCnt]->Mutate();
			}
		}
		
		
		// test
		double error = 0;
		int rightCnt = 0;
		for(int data_i = 0; data_i < test_N; data_i++)
		{
			network[0]->SetInputValue(testSet[data_i]);
			network[0]->SetOutputValue(testLabels[data_i]);
			
			network[0]->ForwardPropagation();
			
			error += network[0]->CalError();
			if(testLabels[data_i][network[0]->CalMaxLabel()] > 0.5)
			{
				rightCnt++;
			}
		}
		std::cout << "test error : " << error / test_N << "\n";
		std::cout << "test accuracy : " << (double) rightCnt / test_N << "\n\n";
	}
}
