#ifndef _NEURON_H_
#define _NEURON_H_

#include "connection.h"
#include "helper.h"
#include <cmath>
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
		
		double value; // this is the value before apply active function
		double activeValue; // this is the value after apply active function
		double trueValue;
		double gradient; // store the gradient of value, not activeValue
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
		double CalError();
		void Display();
		
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
