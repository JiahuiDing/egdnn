#ifndef _NEURON_H_
#define _NEURON_H_

#include "connection.h"
#include "helper.h"
#include <cmath>
#include <vector>
#include <set>

namespace EGDNN
{
	class Connection;
	
	class Neuron
	{	
		public:
		enum Type { input = 0, hidden = 1, output = 2};
		int outputTag;
		Type type;
		double bias;
		std::set<Connection *> outConnections;
		std::set<Connection *> inConnections; // inConnection does not contain weight
		
		double value; // this is the value before apply active function
		double activeValue; // this is the value after apply active function
		double trueValue;
		double gradient; // store the gradient of value, not activeValue
		double sumGradient;
		int counter;
		int copyTag;
		bool visited; // only used in Network::Reachable
		
		Neuron(int outputTag, Type type);
		Neuron(Neuron *neuron); // Copy the outputTag, type, bias from another neuron. Cannot copy its connection.
		~Neuron();
		void PropagateValue();
		void CalGradient();
		void UpdateWeight(double learning_rate);
		void ResetState();
		void PropagateCounter();
		void AddOutNeuron(Neuron *neuron);
		void AddOutNeuron(Neuron *neuron, double weight);
		void AddInNeuron(Neuron *neuron);
		double CalError();
		
		bool ContainOutNeuron(Neuron *neuron);
		bool ContainInNeuron(Neuron *neuron);
		
		double Relu(double x);
		double ReluGrad(double x);
		double Sigmoid(double x);
		double SigmoidGrad(double x);
		
		double MeanSquareError(double activeY, double trueY);
		double MeanSquareErrorGrad(double activeY,double trueY);
		double BinaryCrossEntropy(double activeY, double trueY);
		double BinaryCrossEntropyGrad(double activeY, double trueY);
		double MultiCrossEntropy(double activeY, double trueY);
		double MultiCrossEntropyGrad(double activeY, double trueY);
	};
}


#endif
