#include "test.h"

namespace EGDNN
{
	void SimpleNeuralNetwork(std::vector<std::vector<double>> trainingSet, std::vector<std::vector<double>> trainingLabels, int training_N, 
							std::vector<std::vector<double>> testSet, std::vector<std::vector<double>> testLabels, int test_N, 
							int input_N, int output_N, int maxIter, int batchSize, 
							double learning_rate, double velocity_decay, double regularization_l2, double gradientClip)
	{	
		srand(getpid());
		int hidden_N = 30;
		
		Network *network = new Network(learning_rate, velocity_decay, regularization_l2, gradientClip);
		Neuron *input_neurons[input_N];
		Neuron *hidden_neurons[hidden_N];
		Neuron *output_neurons[output_N];
	
		for(int i = 0; i < input_N; i++)
		{
			input_neurons[i] = new Neuron(-1, Neuron::input);
			network->AddInputNeuron(input_neurons[i]);
		}
		for(int i = 0; i < hidden_N; i++)
		{
			hidden_neurons[i] = new Neuron(-1, Neuron::hidden);
			network->AddHiddenNeuron(hidden_neurons[i]);
		}
		for(int i = 0; i < output_N; i++)
		{
			output_neurons[i] = new Neuron(i, Neuron::output);
			network->AddOutputNeuron(output_neurons[i]);
		}
	
		for(int i = 0; i < input_N; i++)
		{
			for(int j = 0; j < hidden_N; j++)
			{
				input_neurons[i]->AddOutNeuron(hidden_neurons[j]);
				hidden_neurons[j]->AddInNeuron(input_neurons[i]);
			}
		}
	
		for(int i = 0; i < hidden_N; i++)
		{
			for(int j = 0; j < output_N; j++)
			{
				hidden_neurons[i]->AddOutNeuron(output_neurons[j]);
				output_neurons[j]->AddInNeuron(hidden_neurons[i]);
			}
		}
		
		// training
		struct timeval start, end;
		gettimeofday(&start, NULL);
		for(int iterCnt = 0; iterCnt < maxIter; iterCnt++)
		{
			int zeroCnt = 0;
			double error = 0;
			int rightCnt = 0;
			for(int data_i = 0; data_i < training_N; data_i++)
			{
				network->SetInputValue(trainingSet[data_i]);
				network->SetOutputValue(trainingLabels[data_i]);
		
				network->ForwardPropagation();
				network->BackPropagation();
				if(data_i % batchSize == 0)
				{
					network->UpdateWeight();
				}
			
				error += network->CalError();
				zeroCnt += network->CalZeroCnt();
				if(trainingLabels[data_i][network->CalMaxLabel()] > 0.5)
				{
					rightCnt++;
				}
			
				if(data_i % 1000 == 0)
				{
					std::cout << "iterCnt : " << iterCnt << "\n";
					std::cout << "data_i : " << data_i << "\n";
					std::cout << "error : " << error / (data_i + 1) << "\n";
					std::cout << "accuracy : " << (double)rightCnt / (data_i + 1) << "\n";
					std::cout << "zeroRate : " << (double)zeroCnt / ((data_i + 1) * hidden_N) << "\n";
				
					gettimeofday(&end, NULL);
					int timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec -start.tv_usec;
					std::cout << "time : " << timeuse / 1000 << " ms\n\n";
					gettimeofday(&start, NULL);
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
		std::cout << "test accuracy : " << (double)rightCnt / test_N << "\n\n";
	}
}
