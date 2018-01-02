#include "neuron.h"
#include <iostream>
using namespace EGDNN;

Neuron::Neuron(int tag, Type type) : tag(tag), type(type)
{
	bias = fRand(-1, 1);
	outConnections.clear();
	inConnections.clear();
}

Neuron::~Neuron()
{
	outConnections.clear();
	inConnections.clear();
}

// calculate the value of all neuron in the network
void Neuron::PropagateValue()
{
	if(type == input) // input neuron
	{
		for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
		{
			it->neuron->value += it->weight * value;
		}
	}
	else if(type == hidden) // hidden neuron
	{
		value += bias;
		value = value > 0 ? value : 0;
	
		if(value > 0)
		{
			for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
			{
				it->neuron->value += it->weight * value;
			}
		}
	}
	else // output neuron
	{
		value += bias;
		value = value > 0 ? value : 0;
	}
}

// calculate the gradient of all parameters in the network and store it, wait for update
void Neuron::CalGradient()
{
	if(type == input) // input neuron
	{
		for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
		{
			it->AddGradient(learning_rate * it->neuron->gradient * value);
		}
	}
	else if(type == hidden) // hidden neuron
	{
		gradient = 0;
		for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
		{
			gradient += it->neuron->gradient * it->weight;
		}
		gradient = value > 0 ? gradient : 0;
		
		sumGradient += gradient;
		for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
		{
			it->AddGradient(it->neuron->gradient * value);
		}
	}
	else // output neuron
	{
		gradient = trueValue - value;
		gradient = value > 0 ? gradient : 0;
		
		sumGradient += gradient;
		for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
		{
			it->AddGradient(it->neuron->gradient * value);
		}
	}
}

// Update outConnections weight and bias by gradient
void Neuron::UpdateWeight()
{
	bias += learning_rate * sumGradient;
	sumGradient = 0;
	for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
	{
		it->UpdateWeight();
	}
}

// Reset state
void Neuron::ResetState()
{
	if(type != input)
	{
		value = 0;
	}
	gradient = 0;
	counter = 0;
}

// Propagate counter
void Neuron::PropagateCounter()
{
	for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
	{
		it->neuron->counter++;
	}
}

// add an output neuron
void Neuron::AddOutNeuron(Neuron *neuron)
{
	outConnections.push_back(Connection(neuron));
}

// add an input neuron
void Neuron::AddInNeuron(Neuron *neuron)
{
	inConnections.push_back(Connection(neuron));
}

// display all states
void Neuron::Display()
{
	std::cout << "neuron " << tag << " : ";
	if(type == input)
	{
		std::cout << "input\n";
	}
	else if(type == hidden)
	{
		std::cout << "hidden\n";
	}
	else
	{
		std::cout << "output\n";
	}
	
	std::cout << "bias : " << bias << "\n";
	
	std::cout << "outConnections : ";
	for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
	{
		std::cout << it->neuron->tag << " " << it->weight << " , ";
	}
	std::cout << "\n";
	
	std::cout << "inConnections : ";
	for(std::vector<Connection>::iterator it = inConnections.begin(); it != inConnections.end(); it++)
	{
		std::cout << it->neuron->tag << " " << it->weight << " , ";
	}
	std::cout << "\n";
	
	std::cout << "value : " << value << " , " << "trueValue : " << trueValue << " , " << "gradient : " << gradient << " , " << "counter : " << counter << "\n\n";
}
