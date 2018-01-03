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
	
	int hidden_N = 100;
	
	//int training_image_N, training_image_szie;
	int test_image_N, test_image_size;
	//uchar **training_dataset = read_mnist_images("mnist/train-images.idx3-ubyte", training_image_N, training_image_szie);
	//uchar *training_labels = read_mnist_labels("mnist/train-labels.idx1-ubyte", training_image_N);
	uchar **test_dataset = read_mnist_images("mnist/t10k-images.idx3-ubyte", test_image_N, test_image_size);
	uchar *test_labels = read_mnist_labels("mnist/t10k-labels.idx1-ubyte", test_image_N);
	
	Neuron *input_neurons[test_image_size];
	for(int i = 0; i < test_image_size; i++)
	{
		input_neurons[i] = new Neuron(i, Neuron::input);
	}
	
	Neuron *output_neurons[10];
	for(int i = 0; i < 10; i++)
	{
		output_neurons[i] = new Neuron(i + test_image_size, Neuron::output);
	}
	
	Neuron *hidden_neurons[hidden_N];
	for(int i = 0; i < hidden_N; i++)
	{
		hidden_neurons[i] = new Neuron(i + test_image_size + 10, Neuron::hidden);
	}
	
	for(int i = 0; i < test_image_size; i++)
	{
		for(int j = 0; j < hidden_N; j++)
		{
			input_neurons[i]->AddOutNeuron(hidden_neurons[j]);
			hidden_neurons[j]->AddInNeuron(input_neurons[i]);
		}
	}
	
	for(int i = 0; i < hidden_N; i++)
	{
		for(int j = 0; j < 10; j++)
		{
			hidden_neurons[i]->AddOutNeuron(output_neurons[j]);
			output_neurons[j]->AddInNeuron(hidden_neurons[i]);
		}
	}
	
	Network network;
	for(int i = 0; i < test_image_size; i++)
	{
		network.AddNeuron(input_neurons[i]);
	}
	for(int i = 0; i < 10; i++)
	{
		network.AddNeuron(output_neurons[i]);
	}
	for(int i = 0; i < hidden_N; i++)
	{
		network.AddNeuron(hidden_neurons[i]);
	}
	
	double lastError = 1e9;
	int cnt = 0;
	while(true)
	{
		std::cout << "iteration : " << cnt++ << "\n";
		double error = 0;
		for(int i = 0; i < test_image_N; i++)
		{
			for(int j = 0; j < test_image_size; j++)
			{
				input_neurons[j]->value = (double)test_dataset[i][j] / 256;
			}
			
			for(int j = 0; j < 10; j++)
			{
				if((int)test_labels[i] == j)
				{
					output_neurons[j]->trueValue = 1;
				}
				else
				{
					output_neurons[j]->trueValue = 0;
				}
			}
			
			network.ForwardPropagation();
			network.BackPropagation();
			network.UpdateWeight();
			
			double tmperror = network.CalError();
			error += tmperror;
			
			std::cout << i << " " << tmperror << "\n";
		}
		
		std::cout << " " << error / test_image_N << "\n";
		if(fabs(lastError - error) < 1e-12)
		{
			break;
		}
		lastError = error;
		
		// getchar();
	}
	
	return 0;
}
