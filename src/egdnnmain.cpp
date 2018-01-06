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
	
	int data_N;
	int input_N;
	int hidden_N = 30;
	int output_N = 10;
	uchar **dataset = read_mnist_images("mnist/t10k-images.idx3-ubyte", data_N, input_N);
	uchar *labels = read_mnist_labels("mnist/t10k-labels.idx1-ubyte", data_N);
	
	std::cout << (int)labels[15] << "\n";
	for(int i = 0; i < 28; i++)
	{
		for(int j = 0; j < 28; j++)
		{
			if((int)dataset[15][i * 28 + j] > 0)
				std::cout << 1;
			else
				std::cout << 0;
		}
		std::cout << "\n";
		//std::cout << (int)labels[i] << "\n";
	}
	getchar();
	//uchar dataset[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
	//uchar labels[4] = { 0, 1, 1, 0};
	
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
	
	int iterCnt = 0;
	int rightCnt = 0;
	while(true)
	{
		std::cout << "iter : " << ++iterCnt << "\n";
		
		int choose_data = rand() % data_N;
		for(int i = 0; i < input_N; i++)
		{
			input_neurons[i]->value = (double)dataset[choose_data][i];
		}
		for(int i = 0; i < output_N; i++)
		{
			output_neurons[i]->trueValue = 0;
		}
		output_neurons[(int)labels[choose_data]]->trueValue = 1;
		
		network.ForwardPropagation();
		network.BackPropagation();
		network.UpdateWeight();
		
		std::cout << "trueValue : \t";
		for(int i = 0; i < output_N; i++)
		{
			std::cout << output_neurons[i]->trueValue << " , ";
		}
		std::cout << "\n";
		
		std::cout << "activeValue : \t";
		for(int i = 0; i < output_N; i++)
		{
			std::cout << output_neurons[i]->activeValue << " , ";
		}
		std::cout << "\n";
		
		std::cout << "error : " << network.CalError() << "\n";
		
		if(network.CalMaxLabel() == (int)labels[choose_data])
		{
			std::cout << "right\t";
			rightCnt++;
		}
		else
		{
			std::cout << "wrong\t";
		}
		std::cout << "rate : " << (double)rightCnt / iterCnt << "\n\n";
		//getchar();
	}
	
	return 0;
}
