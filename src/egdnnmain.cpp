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
	
	int data_N = 4;
	double data_X[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
	double data_Y[4][1] = { {0}, {1}, {1}, {0} };
	
	int input_N = 2;
	int hidden_N = 4;
	int output_N = 1;
	
	Neuron *input_neurons[input_N];
	Neuron *hidden_neurons[hidden_N];
	Neuron *output_neurons[output_N];
	
	for(int i = 0; i < input_N; i++)
	{
		input_neurons[i] = new Neuron(i, Neuron::input);
	}
	for(int i = 0; i < hidden_N; i++)
	{
		hidden_neurons[i] = new Neuron(i + input_N, Neuron::hidden);
	}
	for(int i = 0; i < output_N; i++)
	{
		output_neurons[i] = new Neuron(i + input_N + hidden_N, Neuron::output);
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
	while(true)
	{
		std::cout << "iter : " << ++iterCnt << "\n";
		for(int i = 0; i < data_N; i++)
		{
			int choose_data = rand() % data_N;
			for(int j = 0; j < input_N; j++)
			{
				input_neurons[j]->value = data_X[choose_data][j];
			}
			for(int j = 0; j < output_N; j++)
			{
				output_neurons[j]->trueValue = data_Y[choose_data][j];
			}
			
			network.ForwardPropagation();
			network.BackPropagation();
			network.UpdateWeight();
			
			//network.Display();
			
			std::cout << "trueValue : " << output_neurons[0]->trueValue << " , ";
			std::cout << "activeValue : " << output_neurons[0]->activeValue << "\n";
			//getchar();
		}
		std::cout << "error : " << network.CalError() << "\n";
		//getchar();
	}
	
	return 0;
}
