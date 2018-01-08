#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "neuron.h"
#include <vector>
#include <set>

namespace EGDNN
{
	class Network
	{
		public:
		std::vector<Neuron *> input_neurons; // input neruons will not be added or deleted, so use vector
		std::set<Neuron *> hidden_neurons; // hidden neurons may be added or deleted, so use set
		std::vector<Neuron *> output_neurons; // output neurons will not be added deleted, so use vector
		
		Network();
		~Network();
		void ForwardPropagation();
		void BackPropagation();
		void UpdateWeight();
		void Mutate();
		void Softmax();
		double CalError();
		int CalZeroCnt(); // calculate the number of hidden neurons whose activeValue = 0
		int NeuronSize(); // return the number of hidden neurons in the network
		int ConnectionSize(); // return the number of connections in the network
		int CalMaxLabel();
		
		bool Reachable(Neuron *s, Neuron *t);
		
		void AddInputNeuron(Neuron *neuron);
		void AddHiddenNeuron(Neuron *neuron);
		void AddOutputNeuron(Neuron *neuron);
		void SetInputValue(std::vector<double> input_values);
		void SetOutputValue(std::vector<double> output_values);
		
		Network *copy();
		void Display();
	};
}

#endif
