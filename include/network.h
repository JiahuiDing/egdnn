#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "neuron.h"
#include <vector>
#include <set>
#include <limits>

namespace EGDNN
{
	class Network
	{
		public:
		double learning_rate;
		double velocity_decay;
		double regularization_l1;
		double regularization_l2;
		double rmsprop_rho;
		double gradientClip;
		std::vector<Neuron *> input_neurons; // input neruons will not be added or deleted, so use vector
		std::set<Neuron *> hidden_neurons; // hidden neurons may be added or deleted, so use set
		std::vector<Neuron *> output_neurons; // output neurons will not be added or deleted, so use vector
		
		Network(double learning_rate, double velocity_decay, double regularization_l1, double regularization_l2, double rmsprop_rho, double gradientClip);
		~Network();
		void ForwardPropagation();
		void BackPropagation();
		void UpdateWeight();
		void Mutate();
		void Eliminate();
		void Softmax();
		double CalError();
		int CalZeroCnt(); // calculate the number of hidden neurons whose activeValue = 0
		int CalNeuronNum(); // calculate the number of hidden neurons in the network
		int CalConnectionNum(); // calculate the number of connections in the network
		double CalAverageWeight(); // calculate the average of all connection weights in the network
		int CalMaxLabel();
		double CalCertainty();
		
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
