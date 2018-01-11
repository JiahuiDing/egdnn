#include "connection.h"
using namespace EGDNN;

Connection::Connection(Neuron *neuron) : neuron(neuron)
{
	weight = fRand(0, 1e-8);
	sumGradient = 0;
}

Connection::Connection(Neuron *neuron, double weight) : neuron(neuron), weight(weight)
{
	sumGradient = 0;
}

void Connection::AddGradient(double gradient)
{
	sumGradient += gradient;
}

void Connection::UpdateWeight(double learning_rate)
{
	weight += learning_rate * sumGradient;
	sumGradient = 0;
}
