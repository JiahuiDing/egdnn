#include "network.h"
#include <queue>
#include <iostream>
using namespace EGDNN;

Network::Network()
{
	input_neurons.clear();
	hidden_neurons.clear();
	output_neurons.clear();
}

Network::~Network()
{	
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		delete (*it);
	}
	input_neurons.clear();
	
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		delete (*it);
	}
	hidden_neurons.clear();
	
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		delete (*it);
	}
	output_neurons.clear();
}

void Network::ForwardPropagation()
{
	// clear all the state
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->ResetState();
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->ResetState();
	}
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->ResetState();
	}
	
	// calculate counter, used for topological sorting
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->PropagateCounter();
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->PropagateCounter();
	}
	
	// perform topological sorting to forward propagate all the value
	std::queue<Neuron *> readyNeurons;
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		readyNeurons.push(neuron);
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->counter == 0)
		{
			readyNeurons.push(neuron);
		}
	}
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->counter == 0)
		{
			readyNeurons.push(neuron);
		}
	}
	
	while(!readyNeurons.empty())
	{
		Neuron *neuron = readyNeurons.front();
		readyNeurons.pop();
		neuron->PropagateValue();
		for(std::set<Connection *>::iterator it = neuron->outConnections.begin(); it != neuron->outConnections.end(); it++)
		{
			Neuron *outNeuron = (*it)->neuron;
			outNeuron->counter--;
			if(outNeuron->counter == 0)
			{
				readyNeurons.push(outNeuron);
			}
		}
	}
	
	//Softmax();
}

void Network::BackPropagation()
{
	// calculate counter, used for topological sorting
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->counter = neuron->outConnections.size();
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->counter = neuron->outConnections.size();
	}
	
	// perform topological sorting to back propagate all the value
	std::queue<Neuron *> readyNeurons;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		readyNeurons.push(neuron);
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->counter == 0)
		{
			readyNeurons.push(neuron);
		}
	}
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->counter == 0)
		{
			readyNeurons.push(neuron);
		}
	}
	
	while(!readyNeurons.empty())
	{
		Neuron *neuron = readyNeurons.front();
		readyNeurons.pop();
		neuron->CalGradient();
		for(std::set<Connection *>::iterator it = neuron->inConnections.begin(); it != neuron->inConnections.end(); it++)
		{
			Neuron *inNeuron = (*it)->neuron;
			inNeuron->counter--;
			if(inNeuron->counter == 0)
			{
				readyNeurons.push(inNeuron);
			}
		}
	}
}

void Network::UpdateWeight()
{
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->UpdateWeight();
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->UpdateWeight();
	}
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->UpdateWeight();
	}
}

void Network::Mutation()
{
}

void Network::Softmax()
{
	double maxValue = 0;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->value > maxValue)
		{
			maxValue = neuron->value;
		}
	}
	
	double sumExpValue = 0;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->activeValue = neuron->value - maxValue;
		sumExpValue += exp(neuron->activeValue);
	}
	
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->activeValue = exp(neuron->activeValue) / sumExpValue;
	}
}

double Network::CalError()
{
	double error = 0;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		error += neuron->CalError();
	}
	return error;
}

int Network::CalZeroCnt() // calculate the number of hidden neurons whose activeValue = 0
{
	int zeroCnt = 0;
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(fabs(neuron->activeValue) < eps)
		{
			zeroCnt++;
		}
	}
	return zeroCnt;
}

int Network::CalMaxLabel()
{
	double maxValue = -1;
	int maxLabel = -1;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->activeValue > maxValue)
		{
			maxValue = neuron->activeValue;
			maxLabel = neuron->outputTag;
		}
	}
	return maxLabel;
}

void Network::AddInputNeuron(Neuron *neuron)
{
	input_neurons.push_back(neuron);
}

void Network::AddHiddenNeuron(Neuron *neuron)
{
	hidden_neurons.insert(neuron);
}

void Network::AddOutputNeuron(Neuron *neuron)
{
	output_neurons.push_back(neuron);
}

void Network::SetInputValue(std::vector<double> input_values)
{
	int N = input_values.size();
	for(int i = 0; i < N; i++)
	{
		input_neurons[i]->value = input_values[i];
	}
}

void Network::SetOutputValue(std::vector<double> output_values)
{
	int N = output_values.size();
	for(int i = 0; i < N; i++)
	{
		output_neurons[i]->trueValue = output_values[i];
	}
}

Network * Network::copy()
{
	int input_N = input_neurons.size();
	int hidden_N = hidden_neurons.size();
	int output_N = output_neurons.size();
	
	Network *new_network = new Network(); // initialize an empty network
	Neuron *new_input_neurons[input_N];
	Neuron *new_hidden_neurons[hidden_N];
	Neuron *new_output_neurons[output_N];
	
	// set copy tag, used to copy connections
	int cnt = 0;
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->copyTag = cnt;
		new_input_neurons[cnt] = new Neuron(neuron);
		new_network->AddInputNeuron(new_input_neurons[cnt]);
		cnt++;
	}
	
	cnt = 0;
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->copyTag = cnt;
		new_hidden_neurons[cnt] = new Neuron(neuron);
		new_network->AddHiddenNeuron(new_hidden_neurons[cnt]);
		cnt++;
	}
	
	cnt = 0;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->copyTag = cnt;
		new_output_neurons[cnt] = new Neuron(neuron);
		new_network->AddOutputNeuron(new_output_neurons[cnt]);
		cnt++;
	}
	
	// copy connections
	for(std::vector<Neuron *>::iterator it1 = input_neurons.begin(); it1 != input_neurons.end(); it1++)
	{
		Neuron *neuron1 = *it1;
		int copyTag1 = neuron1->copyTag;
		for(std::set<Connection *>::iterator it2 = neuron1->outConnections.begin(); it2 != neuron1->outConnections.end(); it2++)
		{
			Neuron *neuron2 = (*it2)->neuron;
			int copyTag2 = neuron2->copyTag;
			double weight = (*it2)->weight;
			new_input_neurons[copyTag1]->AddOutNeuron(new_hidden_neurons[copyTag2], weight);
			new_hidden_neurons[copyTag2]->AddInNeuron(new_input_neurons[copyTag1]);
		}
	}
	
	for(std::set<Neuron *>::iterator it1 = hidden_neurons.begin(); it1 != hidden_neurons.end(); it1++)
	{
		Neuron *neuron1 = *it1;
		int copyTag1 = neuron1->copyTag;
		for(std::set<Connection *>::iterator it2 = neuron1->outConnections.begin(); it2 != neuron1->outConnections.end(); it2++)
		{
			Neuron *neuron2 = (*it2)->neuron;
			int copyTag2 = neuron2->copyTag;
			double weight = (*it2)->weight;
			
			if(neuron2->type == Neuron::hidden)
			{
				new_hidden_neurons[copyTag1]->AddOutNeuron(new_hidden_neurons[copyTag2], weight);
				new_hidden_neurons[copyTag2]->AddInNeuron(new_hidden_neurons[copyTag1]);
			}
			else
			{
				new_hidden_neurons[copyTag1]->AddOutNeuron(new_output_neurons[copyTag2], weight);
				new_output_neurons[copyTag2]->AddInNeuron(new_hidden_neurons[copyTag1]);
			}
		}
	}
	
	return new_network;
}

void Network::Display()
{
}
