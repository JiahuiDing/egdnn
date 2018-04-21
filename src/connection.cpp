#include "connection.h"
using namespace EGDNN;

Connection::Connection(Neuron *inNeuron, Neuron *outNeuron) : inNeuron(inNeuron), outNeuron(outNeuron)
{
	weight = fRand(-0.05, 0.05);
	velocity = 0;
	sumGradient = 0;
	rmsprop_s = 0;
}

Connection::Connection(Neuron *inNeuron, Neuron *outNeuron, double weight) : inNeuron(inNeuron), outNeuron(outNeuron), weight(weight)
{
	velocity = 0;
	sumGradient = 0;
	rmsprop_s = 0;
}

void Connection::AddGradient(double gradient)
{
	sumGradient += gradient;
}

void Connection::UpdateWeight(double learning_rate, double velocity_decay, double regularization_l1, double regularization_l2, double rmsprop_rho)
{
	if(rmsprop_rho < 0)
	{
		velocity = velocity_decay * velocity + sumGradient;
		sumGradient = 0;
		weight = weight + learning_rate * velocity - learning_rate * regularization_l1 * fabs(weight) / weight - learning_rate * regularization_l2 * weight;
	}
	else
	{
		rmsprop_s = rmsprop_rho * rmsprop_s + (1 - rmsprop_rho) * sumGradient * sumGradient;
		weight = weight + learning_rate * sumGradient / sqrt(rmsprop_s + 1e-6) - learning_rate * regularization_l1 * fabs(weight) / weight - learning_rate * regularization_l2 * weight;
		sumGradient = 0;
	}
}
