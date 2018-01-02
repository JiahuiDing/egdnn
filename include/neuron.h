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
		int tag;
		Type type;
		double bias;
		std::vector<Connection> outConnections;
		std::vector<Connection> inConnections; // inConnection does not contain weight
		
		double value;
		double trueValue;
		double gradient;
		double sumGradient;
		int counter;
		
		Neuron(int tag, Type type);
		~Neuron();
		void PropagateValue();
		void CalGradient();
		void UpdateWeight();
		void ResetState();
		void PropagateCounter();
		void AddOutNeuron(Neuron *neuron);
		void AddInNeuron(Neuron *neuron);
		void Display();
	};
}


#endif
