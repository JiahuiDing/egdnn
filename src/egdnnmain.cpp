#include "egdnnmain.h"
#include "neuron.h"
#include "connection.h"
#include "network.h"
#include "helper.h"
#include <iostream>
#include <cstdio>
#include <unistd.h>
#include <cmath>
using namespace EGDNN;

int main(int argc, char *argv[])
{
	srand(getpid());
	
	int batch_size = 100; // update weight and bias after seeing batch_size number of datas
	int training_data_N;
	int test_data_N;
	int input_N;
	int hidden_N = 300;
	int output_N = 10;
	
	uchar **training_dataset = read_mnist_images("mnist/train-images.idx3-ubyte", training_data_N, input_N);
	uchar *training_labels = read_mnist_labels("mnist/train-labels.idx1-ubyte", training_data_N);
	uchar **test_dataset = read_mnist_images("mnist/t10k-images.idx3-ubyte", test_data_N, input_N);
	uchar *test_labels = read_mnist_labels("mnist/t10k-labels.idx1-ubyte", test_data_N);
	
	Neuron *input_neurons[input_N];
	Neuron *hidden_neurons[hidden_N];
	Neuron *output_neurons[output_N];
	
	for(int i = 0; i < input_N; i++)
	{
		input_neurons[i] = new Neuron(-1, Neuron::input);
	}
	for(int i = 0; i < hidden_N; i++)
	{
		hidden_neurons[i] = new Neuron(-1, Neuron::hidden);
	}
	for(int i = 0; i < output_N; i++)
	{
		output_neurons[i] = new Neuron(i, Neuron::output);
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
	
	Network network;
	for(int i = 0; i < input_N; i++)
	{
		network.AddNeuron(input_neurons[i]);
	}
	for(int i = 0; i < hidden_N; i++)
	{
		network.AddNeuron(hidden_neurons[i]);
	}
	for(int i = 0; i < output_N; i++)
	{
		network.AddNeuron(output_neurons[i]);
	}
	
	// training
	int trainingIterCnt = 0;
	double lastError = 1e10;
	while(true)
	{
		trainingIterCnt++;
		double error = 0;
		int rightCnt = 0;
		for(int data_i = 0; data_i < training_data_N; data_i++)
		{
			for(int i = 0; i < input_N; i++)
			{
				input_neurons[i]->value = (double)training_dataset[data_i][i] / 255;
			}
			for(int i = 0; i < output_N; i++)
			{
				output_neurons[i]->trueValue = 0;
			}
			output_neurons[(int)training_labels[data_i]]->trueValue = 1;
		
			network.ForwardPropagation();
			network.BackPropagation();
			if(data_i % batch_size == 0)
			{
				network.UpdateWeight();
			}
			
			error += network.CalError();
			if(network.CalMaxLabel() == (int)training_labels[data_i])
			{
				rightCnt++;
			}
			
			if(data_i % 1000 == 0)
			{
				std::cout << "trainingIterCnt : " << trainingIterCnt << "\n";
				std::cout << "data_i : " <<data_i << "\n";
				std::cout << "error : " << error / (data_i + 1) << "\n";
				std::cout << "accuracy : " << (double)rightCnt / (data_i + 1) << "\n\n";
			}
		}
		error /= training_data_N;
		if(fabs(lastError - error) < 1e-8)
		{
			break;
		}
		lastError = error;
	}
	
	// test
	double error = 0;
	int rightCnt = 0;
	for(int data_i = 0; data_i < test_data_N; data_i++)
	{
		for(int i = 0; i < input_N; i++)
		{
			input_neurons[i]->value = (double)test_dataset[data_i][i] / 255;
		}
		for(int i = 0; i < output_N; i++)
		{
			output_neurons[i]->trueValue = 0;
		}
		output_neurons[(int)test_labels[data_i]]->trueValue = 1;
		
		network.ForwardPropagation();
		
		error += network.CalError();
		if(network.CalMaxLabel() == (int)test_labels[data_i])
		{
			rightCnt++;
		}
	}
	std::cout << "test error : " << error / test_data_N << "\n";
	std::cout << "test accuracy : " << (double)rightCnt / test_data_N << "\n\n";
	
	return 0;
}
