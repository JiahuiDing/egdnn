#ifndef _CONNECTION_H_
#define _CONNECTION_H_

#include "neuron.h"

namespace EGDNN
{
	class Neuron;
	
	class Connection
	{
		public:
		Neuron *inNeuron;
		Neuron *outNeuron;
		double weight;
		double velocity;
		double sumGradient; // store the sum gradient of a batch
		
		Connection(Neuron *inNeuron, Neuron *outNeuron);
		Connection(Neuron *inNeuron, Neuron *outNeuron, double weight);
		void AddGradient(double gradient);
		void UpdateWeight(double learning_rate, double velocity_decay, double regularization_l2);
	};
}

#endif
