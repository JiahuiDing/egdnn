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
		for(std::vector<Connection>::iterator it = neuron->outConnections.begin(); it != neuron->outConnections.end(); it++)
		{
			Neuron *outNeuron = it->neuron;
			outNeuron->counter--;
			if(outNeuron->counter == 0)
			{
				readyNeurons.push(outNeuron);
			}
		}
	}
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
		for(std::vector<Connection>::iterator it = neuron->inConnections.begin(); it != neuron->inConnections.end(); it++)
		{
			Neuron *inNeuron = it->neuron;
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
