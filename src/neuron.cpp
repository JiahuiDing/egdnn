#include "neuron.h"
#include <iostream>
using namespace EGDNN;

Neuron::Neuron(int outputTag, Type type) : outputTag(outputTag), type(type)
{
	bias = fRand(-0.05, 0.05);
	outConnections.clear();
	inConnections.clear();
	
	value = 0;
	activeValue = 0;
	trueValue = 0;
	gradient = 0;
	rmsprop_s = 0;
	velocity = 0;
	sumGradient = 0;
	forwardCounter = 0;
	backwardCounter = 0;
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
	rmsprop_s = 0;
	velocity = 0;
	sumGradient = 0;
	forwardCounter = 0;
	backwardCounter = 0;
}

Neuron::~Neuron()
{
	for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
	{
		delete (*it);
	}
	outConnections.clear();
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
				(*it)->outNeuron->value += (*it)->weight * activeValue;
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
				(*it)->outNeuron->value += (*it)->weight * activeValue;
			}
		}
	}
	else // output neuron
	{
		value += bias;
		activeValue = value; // linear activation function, used for regression
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
				(*it)->AddGradient((*it)->outNeuron->gradient * activeValue);
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
				gradient += (*it)->outNeuron->gradient * (*it)->weight;
			}
			gradient *= ReluGrad(value);
			sumGradient += gradient;
		
			for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
			{
				(*it)->AddGradient((*it)->outNeuron->gradient * activeValue);
			}
		}
	}
	else // output neuron
	{
		gradient = - MeanSquareErrorGrad(activeValue, trueValue);
		//gradient = - SoftmaxCrossEntropyGrad(activeValue, trueValue);
		sumGradient += gradient;
	}
}

// Update outConnections weight and bias by gradient
void Neuron::UpdateWeight(double learning_rate, double velocity_decay, double regularization_l1, double regularization_l2, double rmsprop_rho)
{	
	if(rmsprop_rho < 0)
	{
		velocity = velocity_decay * velocity + sumGradient;
		sumGradient = 0;
		bias = bias + learning_rate * velocity;
	}
	else
	{
		rmsprop_s = rmsprop_rho * rmsprop_s + (1 - rmsprop_rho) * sumGradient * sumGradient;
		bias = bias + learning_rate * sumGradient / sqrt(rmsprop_s + 1e-6);
		sumGradient = 0;
	}
	
	for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
	{
		(*it)->UpdateWeight(learning_rate, velocity_decay, regularization_l1, regularization_l2, rmsprop_rho);
	}
}

// Reset states except sumGradient and velocity, set forwardCounter and backwardCounter
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
	
	forwardCounter = inConnections.size();
	backwardCounter = outConnections.size();
}

// add an output connection
void Neuron::AddOutConnection(Connection *connection)
{
	outConnections.insert(connection);
}

// add an input connection
void Neuron::AddInConnection(Connection *connection)
{
	inConnections.insert(connection);
}

double Neuron::CalError()
{
	return type == output ? MeanSquareError(activeValue, trueValue) : 0;
}

bool Neuron::ContainOutNeuron(Neuron *neuron)
{
	for(std::set<Connection *>::iterator it = outConnections.begin(); it != outConnections.end(); it++)
	{
		if((*it)->outNeuron == neuron)
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
		if((*it)->inNeuron == neuron)
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

// The gradient of Softmax + Cross-entropy error
double Neuron::SoftmaxCrossEntropyGrad(double activeY,double trueY)
{
	return activeY - trueY;
}
