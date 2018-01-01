#ifndef _NEURON_H_
#define _NEURON_H_

#include "connection.h"
#include "helper.h"
#include <vector>

namespace EGDNN
{
	class Connection;
	
	class Neuron
	{	
		public:
		enum Type { input = 0, hidden = 1, output = 2};
		Type type;
		int tag;
		double bias;
		std::vector<Connection> outConnections;
		std::vector<Connection> inConnections; // inConnection does not contain weight
		
		double value;
		double trueValue;
		double gradient;
		int counter;
		
		Neuron(Type type, int tag);
		~Neuron();
		void PropagateValue();
		void CalGradient();
		void UpdateByGradient();
		void ResetState();
		void PropagateCounter();
		void AddOutNeuron(Neuron *neuron);
		void AddInNeuron(Neuron *neuron);
		void Display();
	};
}


#endif
