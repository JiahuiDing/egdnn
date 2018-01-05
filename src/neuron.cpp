#include "neuron.h"
#include <iostream>
using namespace EGDNN;

Neuron::Neuron(int tag, Type type) : tag(tag), type(type)
{
	bias = fRand(0, 1);
	outConnections.clear();
	inConnections.clear();
	
	value = 0;
	activeValue = 0;
	trueValue = 0;
	gradient = 0;
	sumGradient = 0;
	counter = 0;
}

Neuron::~Neuron()
{
	outConnections.clear();
	inConnections.clear();
}

// calculate the value and activeValue of all neuron in the network
void Neuron::PropagateValue()
{
	if(type == input) // input neuron
	{
		activeValue = value;
		if(fabs(activeValue) > eps)
		{
			for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
			{
				it->neuron->value += it->weight * activeValue;
			}
		}
	}
	else if(type == hidden) // hidden neuron
	{
		value += bias;
		activeValue = Relu(value);
	
		if(fabs(activeValue) > eps)
		{
			for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
			{
				it->neuron->value += it->weight * activeValue;
			}
		}
	}
	else // output neuron
	{
		value += bias;
		activeValue = Relu(value);
	}
}

// calculate the gradient of all parameters in the network and store it, wait for update
void Neuron::CalGradient()
{
	if(type == input) // input neuron
	{
		if(fabs(activeValue) > eps)
		{
			for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
			{
				it->AddGradient(it->neuron->gradient * activeValue);
			}
		}
	}
	else if(type == hidden) // hidden neuron
	{
		gradient = 0;
		for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
		{
			gradient += it->neuron->gradient * it->weight;
		}
		gradient *= ReluGrad(value);
		sumGradient += gradient;
		
		for(std::vector<Connection>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
		{
			it->AddGradient(it->neuron->gradient * activeValue);
		}
	}
	else // output neuron
	{
		gradient = - MeanSquareErrorGrad(activeValue, trueValue) * ReluGrad(value);
		sumGradient += gradient;
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

// Reset state except sumGradient
void Neuron::ResetState()
{
	if(type != input)
	{
		value = 0;
	}
	activeValue = 0;
	if(type != output)
	{
		trueValue = 0;
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

double Neuron::CalError()
{
	return type == output ? MeanSquareError(activeValue, trueValue) : 0;
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


double Neuron::Relu(double x)
{
	return x > eps ? x : 0;
}

double Neuron::ReluGrad(double x)
{
	return x > eps ? 1 : 0;
}

double Neuron::Sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

double Neuron::SigmoidGrad(double x)
{
	double fx = Sigmoid(x);
	return fx * (1 - fx);
}

double Neuron::MeanSquareError(double activeY, double trueY)
{
	return 0.5 * (activeY - trueY) * (activeY - trueY);
}

double Neuron::MeanSquareErrorGrad(double activeY, double trueY)
{
	return activeY - trueY;
}

double Neuron::BinaryCrossEntropy(double activeY, double trueY)
{
	if(trueY == 0)
	{
		return -log(1 - activeY);
	}
	else
	{
		return -log(activeY);
	}
}

double Neuron::BinaryCrossEntropyGrad(double activeY, double trueY)
{
	if(trueY == 0)
	{
		return 1 / (1 - activeY);
	}
	else
	{
		return - 1 / activeY;
	}
}
