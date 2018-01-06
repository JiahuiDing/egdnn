#include "network.h"
#include <queue>
#include <iostream>
using namespace EGDNN;

Network::Network()
{
	neurons.clear();
}

Network::~Network()
{	
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
	{
		delete (*it);
	}
	neurons.clear();
}

void Network::ForwardPropagation()
{
	// clear all the state
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->ResetState();
	}
	
	// calculate counter, used for topological sorting
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->PropagateCounter();
	}
	
	// perform topological sorting to forward propagate all the value
	std::queue<Neuron *> readyNeurons;
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
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
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->counter = neuron->outConnections.size();
	}
	
	// perform topological sorting to back propagate all the value
	std::queue<Neuron *> readyNeurons;
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
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
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->UpdateWeight();
	}
}

void Network::Softmax()
{
	double maxValue = 0;
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->type == Neuron::output)
		{
			if(neuron->value > maxValue)
			{
				maxValue = neuron->value;
			}
		}
	}
	
	double sumExpValue = 0;
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->type == Neuron::output)
		{
			neuron->value -= maxValue;
			sumExpValue += exp(neuron->value);
		}
	}
	
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->type == Neuron::output)
		{
			neuron->activeValue = exp(neuron->value) / sumExpValue;
		}
	}
}

double Network::CalError()
{
	double error = 0;
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->type == Neuron::output)
		{
			error += neuron->CalError();
		}
	}
	return error;
}

int Network::CalMaxLabel()
{
	double maxValue = -1;
	int maxLabel = -1;
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->type == Neuron::output)
		{
			if(neuron->activeValue > maxValue)
			{
				maxValue = neuron->activeValue;
				maxLabel = neuron->tag;
			}
		}
	}
	return maxLabel;
}

void Network::AddNeuron(Neuron *neuron)
{
	neurons.push_back(neuron);
}

void Network::Display()
{
	for(std::vector<Neuron *>::iterator it = neurons.begin(); it != neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->Display();
	}
}
