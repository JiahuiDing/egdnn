#include "neuron.h"
#include <iostream>
using namespace EGDNN;

Neuron::Neuron(int outputTag, Type type) : outputTag(outputTag), type(type)
{
	bias = fRand(0, 1);
	outConnections.clear();
	inConnections.clear();
	
	value = 0;
	activeValue = 0;
	trueValue = 0;
	gradient = 0;
	sumGradient = 0;
	sumGradientCnt = 0;
	counter = 0;
}

// Copy the outputTag, type, bias from another neuron. Cannot copy its connection.
Neuron::Neuron(Neuron *neuron) : outputTag(neuron->outputTag), type(neuron->type), bias(neuron->bias)
{
	outConnections.clear();
	inConnections.clear();
	
	value = 0;
	activeValue = 0;
	trueValue = 0;
	gradient = 0;
	sumGradient = 0;
	sumGradientCnt = 0;
	counter = 0;
}

Neuron::~Neuron()
{
	for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
	{
		delete (*it);
	}
	outConnections.clear();
	
	for(std::set<Connection *>::iterator it = inConnections.begin(); it != inConnections.end(); it++)
	{
		delete (*it);
	}
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
			for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
			{
				(*it)->neuron->value += (*it)->weight * activeValue;
			}
		}
	}
	else if(type == hidden) // hidden neuron
	{
		value += bias;
		activeValue = Relu(value);
	
		if(fabs(activeValue) > eps)
		{
			for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
			{
				(*it)->neuron->value += (*it)->weight * activeValue;
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
			for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
			{
				(*it)->AddGradient((*it)->neuron->gradient * activeValue);
			}
		}
	}
	else if(type == hidden) // hidden neuron
	{
		gradient = 0;
		
		if(fabs(activeValue) > eps)
		{
			for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
			{
				gradient += (*it)->neuron->gradient * (*it)->weight;
			}
			gradient *= ReluGrad(value);
			sumGradient += gradient;
			sumGradientCnt++;
		
			for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
			{
				(*it)->AddGradient((*it)->neuron->gradient * activeValue);
			}
		}
	}
	else // output neuron
	{
		gradient = - MeanSquareErrorGrad(activeValue, trueValue) * ReluGrad(value);
		//gradient = - MultiCrossEntropyGrad(activeValue, trueValue) * activeValue * (1 - activeValue);
		sumGradient += gradient;
		sumGradientCnt++;
	}
}

// Update outConnections weight and bias by gradient
void Neuron::UpdateWeight(double learning_rate)
{
	bias += learning_rate * sumGradient / sumGradientCnt;
	sumGradient = 0;
	sumGradientCnt = 0;
	for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
	{
		(*it)->UpdateWeight(learning_rate);
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
	for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
	{
		(*it)->neuron->counter++;
	}
}

// add an output neuron
void Neuron::AddOutNeuron(Neuron *neuron)
{
	outConnections.insert(new Connection(neuron));
}

// add an output neuron
void Neuron::AddOutNeuron(Neuron *neuron, double weight)
{
	outConnections.insert(new Connection(neuron, weight));
}

// add an input neuron
void Neuron::AddInNeuron(Neuron *neuron)
{
	inConnections.insert(new Connection(neuron));
}

double Neuron::CalError()
{
	return type == output ? MeanSquareError(activeValue, trueValue) : 0;
}

bool Neuron::ContainOutNeuron(Neuron *neuron)
{
	for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
	{
		if((*it)->neuron == neuron)
		{
			return true;
		}
	}
	return false;
}

bool Neuron::ContainInNeuron(Neuron *neuron)
{
	for(std::set<Connection *>::iterator it = inConnections.begin(); it != inConnections.end(); it++)
	{
		if((*it)->neuron == neuron)
		{
			return true;
		}
	}
	return false;
}

// Relu
double Neuron::Relu(double x)
{
	return x > eps ? x : 0;
}

// The gradient of Relu
double Neuron::ReluGrad(double x)
{
	return x > eps ? 1 : 0;
}

// Sigmoid
double Neuron::Sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

// The gradient of Sigmoid
double Neuron::SigmoidGrad(double x)
{
	double fx = Sigmoid(x);
	return fx * (1 - fx);
}

// Mean Square Error
double Neuron::MeanSquareError(double activeY, double trueY)
{
	return 0.5 * (activeY - trueY) * (activeY - trueY);
}

// The gradient of Mean Square Error
double Neuron::MeanSquareErrorGrad(double activeY, double trueY)
{
	return activeY - trueY;
}

double Neuron::BinaryCrossEntropy(double activeY, double trueY)
{
	if(abs(trueY) < eps)
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
	if(abs(trueY) < eps)
	{
		return 1 / (1 - activeY);
	}
	else
	{
		return - 1 / activeY;
	}
}

double Neuron::MultiCrossEntropy(double activeY, double trueY)
{
	if(abs(trueY) < eps)
	{
		return 0;
	}
	else
	{
		return - log(activeY);
	}
}
double Neuron::MultiCrossEntropyGrad(double activeY, double trueY)
{
	if(abs(trueY) < eps)
	{
		return 0;
	}
	else
	{
		return - 1 / activeY;
	}
}
