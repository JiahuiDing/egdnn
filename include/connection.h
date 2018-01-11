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
		double sumGradient;
		int sumGradientCnt;
		
		Connection(Neuron *neuron);
		Connection(Neuron *neuron, double weight);
		void AddGradient(double gradient);
		void UpdateWeight(double learning_rate);
	};
}

#endif
