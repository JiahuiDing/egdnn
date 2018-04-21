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
		std::set<Connection *> inConnections;
		
		double value; // this is the value before apply active function
		double activeValue; // this is the value after apply active function
		double trueValue;
		
		double gradient; // store the gradient of value, not activeValue, for a single input. Used for bakcpropagation.
		double rmsprop_s;
		double velocity;
		double sumGradient; // store the gradient of a batch
		
		int forwardCounter;
		int backwardCounter;
		int displayTag; // only used in Network::Display
		int copyTag;
		bool visited; // only used in Network::Reachable
		bool isNew; // only used in Network::Mutate
		
		Neuron(int outputTag, Type type);
		Neuron(Neuron *neuron); // Copy the outputTag, type, bias from another neuron. Cannot copy its connection.
		~Neuron();
		void PropagateValue();
		void CalGradient();
		void UpdateWeight(double learning_rate, double velocity_decay, double regularization_l1, double regularization_l2, double rmsprop_rho);
		void ResetState();
		void AddOutConnection(Connection *connection);
		void AddInConnection(Connection *connection);
		double CalError();
		
		bool ContainOutNeuron(Neuron *neuron);
		bool ContainInNeuron(Neuron *neuron);
		
		double Relu(double x);
		double ReluGrad(double x);
		double Sigmoid(double x);
		double SigmoidGrad(double x);
		
		double MeanSquareError(double activeY, double trueY);
		double MeanSquareErrorGrad(double activeY,double trueY);
		double SoftmaxCrossEntropyGrad(double activeY,double trueY);
	};
}


#endif
