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
		
		Connection(Neuron *neuron);
		void AddGradient(double gradient);
		void UpdateWeight();
	};
}

#endif
