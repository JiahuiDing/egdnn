#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "neuron.h"
#include <vector>

namespace EGDNN
{
	class Network
	{
		public:
		std::vector<Neuron *> neurons;
		
		Network();
		~Network();
		void ForwardPropagation();
		void BackPropagation();
		void UpdateWeight();
		void Softmax();
		double CalError();
		int CalMaxLabel();
		void AddNeuron(Neuron *neuron);
		void Display();
	};
}

#endif
