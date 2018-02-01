#include "connection.h"
using namespace EGDNN;

Connection::Connection(Neuron *inNeuron, Neuron *outNeuron) : inNeuron(inNeuron), outNeuron(outNeuron)
{
	weight = fRand(0, 1e-3);
	velocity = 0;
	sumGradient = 0;
}

Connection::Connection(Neuron *inNeuron, Neuron *outNeuron, double weight) : inNeuron(inNeuron), outNeuron(outNeuron), weight(weight)
{
	velocity = 0;
	sumGradient = 0;
}

void Connection::AddGradient(double gradient)
{
	sumGradient += gradient;
}

void Connection::UpdateWeight(double learning_rate, double velocity_decay, double regularization_l2)
{
	velocity = velocity_decay * velocity + sumGradient;
	sumGradient = 0;
	weight = weight + learning_rate * velocity - learning_rate * regularization_l2 * weight;
}
