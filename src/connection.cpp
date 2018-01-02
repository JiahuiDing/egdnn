#include "connection.h"
using namespace EGDNN;

Connection::Connection(Neuron *neuron) : neuron(neuron)
{
	weight = fRand(-1, 1);
}

void Connection::AddGradient(double gradient)
{
	sumGradient += gradient;
}

void Connection::UpdateWeight()
{
	weight += learning_rate * sumGradient;
	sumGradient = 0;
}
