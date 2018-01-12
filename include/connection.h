#ifndef _CONNECTION_H_
#define _CONNECTION_H_

#include "neuron.h"

namespace EGDNN
{
	class Neuron;
	
	class Connection
	{
		public:
		Neuron *neuron;
		double weight;
		double velocity;
		double sumGradient; // store the sum gradient of a batch
		
		Connection(Neuron *neuron);
		Connection(Neuron *neuron, double weight);
		void AddGradient(double gradient);
		void UpdateWeight(double learning_rate, double velocity_decay);
	};
}

#endif
