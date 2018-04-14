#include "network.h"
#include <queue>
#include <iostream>
using namespace EGDNN;

Network::Network(double learning_rate, double velocity_decay, double regularization_l2, double gradientClip) 
				: learning_rate(learning_rate), velocity_decay(velocity_decay), regularization_l2(regularization_l2), gradientClip(gradientClip)
{
	input_neurons.clear();
	hidden_neurons.clear();
	output_neurons.clear();
}

Network::~Network()
{	
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		delete (*it);
	}
	input_neurons.clear();
	
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		delete (*it);
	}
	hidden_neurons.clear();
	
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		delete (*it);
	}
	output_neurons.clear();
}

void Network::ForwardPropagation()
{
	// reset all the state, set forwardCounter and backwardCounter
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->ResetState();
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->ResetState();
	}
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->ResetState();
	}
	
	// perform topological sorting to forward propagate all the value
	std::queue<Neuron *> readyNeurons;
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		readyNeurons.push(neuron);
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->forwardCounter == 0)
		{
			readyNeurons.push(neuron);
		}
	}
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->forwardCounter == 0)
		{
			readyNeurons.push(neuron);
		}
	}
	
	while(!readyNeurons.empty())
	{
		Neuron *neuron = readyNeurons.front();
		readyNeurons.pop();
		neuron->PropagateValue();
		for(std::set<Connection *>::iterator it = neuron->outConnections.begin(); it != neuron->outConnections.end(); it++)
		{
			Neuron *outNeuron = (*it)->outNeuron;
			outNeuron->forwardCounter--;
			if(outNeuron->forwardCounter == 0)
			{
				readyNeurons.push(outNeuron);
			}
		}
	}
	
	//Softmax();
}

void Network::BackPropagation()
{	
	// perform topological sorting to back propagate all the value
	std::queue<Neuron *> readyNeurons;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		readyNeurons.push(neuron);
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->backwardCounter == 0)
		{
			readyNeurons.push(neuron);
		}
	}
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->backwardCounter == 0)
		{
			readyNeurons.push(neuron);
		}
	}
	
	while(!readyNeurons.empty())
	{
		Neuron *neuron = readyNeurons.front();
		readyNeurons.pop();
		neuron->CalGradient();
		for(std::set<Connection *>::iterator it = neuron->inConnections.begin(); it != neuron->inConnections.end(); it++)
		{
			Neuron *inNeuron = (*it)->inNeuron;
			inNeuron->backwardCounter--;
			if(inNeuron->backwardCounter == 0)
			{
				readyNeurons.push(inNeuron);
			}
		}
	}
}

void Network::UpdateWeight()
{
	double sumGradient = 0;
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		for(std::set<Connection *>::iterator it = neuron->outConnections.begin(); it != neuron->outConnections.end(); it++)
		{
			sumGradient += (*it)->sumGradient * (*it)->sumGradient;
		}
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		sumGradient += neuron->sumGradient * neuron->sumGradient;
		for(std::set<Connection *>::iterator it = neuron->outConnections.begin(); it != neuron->outConnections.end(); it++)
		{
			sumGradient += (*it)->sumGradient * (*it)->sumGradient;
		}
	}
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		sumGradient += neuron->sumGradient * neuron->sumGradient;
	}
	sumGradient = sqrt(sumGradient);
	
	double tmpGradientClip = gradientClip * sqrt(CalConnectionNum() + CalNeuronNum());
	if(sumGradient > tmpGradientClip)
	{
		for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
		{
			Neuron *neuron = *it;
			for(std::set<Connection *>::iterator it = neuron->outConnections.begin(); it != neuron->outConnections.end(); it++)
			{
				(*it)->sumGradient *= tmpGradientClip / sumGradient; 
			}
		}
		for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
		{
			Neuron *neuron = *it;
			neuron->sumGradient *= tmpGradientClip / sumGradient;
			for(std::set<Connection *>::iterator it = neuron->outConnections.begin(); it != neuron->outConnections.end(); it++)
			{
				(*it)->sumGradient *= tmpGradientClip / sumGradient; 
			}
		}
		for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
		{
			Neuron *neuron = *it;
			neuron->sumGradient *= tmpGradientClip / sumGradient;
		}
	}
	
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->UpdateWeight(learning_rate, velocity_decay, regularization_l2);
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->UpdateWeight(learning_rate, velocity_decay, regularization_l2);
	}
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->UpdateWeight(learning_rate, velocity_decay, regularization_l2);
	}
}

void Network::Mutate()
{
	/*
	for(std::vector<Neuron *>::iterator it1 = input_neurons.begin(); it1 != input_neurons.end(); it1++)
	{
		Neuron *neuron1 = *it1;
		for(std::set<Connection *>::iterator it2 = neuron1->outConnections.begin(); it2 != neuron1->outConnections.end(); it2++)
		{
			Connection *connection = *it2;
			connection->weight += fRand(-1, 1);
		}
	}
	for(std::set<Neuron *>::iterator it1 = hidden_neurons.begin(); it1 != hidden_neurons.end(); it1++)
	{
		Neuron *neuron1 = *it1;
		for(std::set<Connection *>::iterator it2 = neuron1->outConnections.begin(); it2 != neuron1->outConnections.end(); it2++)
		{
			Connection *connection = *it2;			
			connection->weight += fRand(-1, 1);
		}
	}
	*/
	
	/*
	int hidden1_N = 32;
	int hidden2_N = 32;
	Neuron * n1[hidden1_N];
	Neuron * n2[hidden2_N];
	
	for(int i = 0; i < hidden1_N; i++)
	{
		n1[i] = new Neuron(-1, Neuron::hidden);
		hidden_neurons.insert(n1[i]);
	}
	
	for(int i = 0; i < hidden2_N; i++)
	{
		n2[i] = new Neuron(-1, Neuron::hidden);
		hidden_neurons.insert(n2[i]);
	}
	
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron1 = *it;
		for(int i = 0; i < hidden1_N; i++)
		{
			Neuron *neuron2 = n1[i];
			Connection *connection = new Connection(neuron1, neuron2);
			neuron1->AddOutConnection(connection);
			neuron2->AddInConnection(connection);
		}
	}
	
	for(int i = 0; i < hidden1_N; i++)
	{
		Neuron *neuron1 = n1[i];
		for(int j = 0; j < hidden2_N; j++)
		{
			Neuron *neuron2 = n2[j];
			Connection *connection = new Connection(neuron1, neuron2);
			neuron1->AddOutConnection(connection);
			neuron2->AddInConnection(connection);
		}
	}
	
	for(int i = 0; i < hidden2_N; i++)
	{
		Neuron *neuron1 = n2[i];
		for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
		{
			Neuron *neuron2 = *it;
			Connection *connection = new Connection(neuron1, neuron2);
			neuron1->AddOutConnection(connection);
			neuron2->AddInConnection(connection);
		}
	}
	*/
	
	int newHiddenNeuronNum = 32;
	double rateInputHidden = 1;
	double rateHiddenHidden = 0;
	double rateHiddenOutput = 1;
	
	// add hidden neurons
	for(int i = 0; i < newHiddenNeuronNum; i++)
	{
		hidden_neurons.insert(new Neuron(-1, Neuron::hidden));
	}
	
	// add new connections between input_neurons and hidden_neurons
	for(std::vector<Neuron *>::iterator it1 = input_neurons.begin(); it1 != input_neurons.end(); it1++)
	{
		Neuron *neuron1 = *it1;
		for(std::set<Neuron *>::iterator it2 = hidden_neurons.begin(); it2 != hidden_neurons.end(); it2++)
		{
			Neuron *neuron2 = *it2;
			if(fRand(0,1) < rateInputHidden && neuron1->ContainOutNeuron(neuron2) == false)
			{
				Connection *connection = new Connection(neuron1, neuron2);
				neuron1->AddOutConnection(connection);
				neuron2->AddInConnection(connection);
			}
		}
	}
	
	// add new connections between hidden_neurons and output_neurons
	for(std::set<Neuron *>::iterator it1 = hidden_neurons.begin(); it1 != hidden_neurons.end(); it1++)
	{
		Neuron *neuron1 = *it1;
		for(std::vector<Neuron *>::iterator it2 = output_neurons.begin(); it2 != output_neurons.end(); it2++)
		{
			Neuron *neuron2 = *it2;
			if(fRand(0,1) < rateHiddenOutput && neuron1->ContainOutNeuron(neuron2) == false)
			{
				Connection *connection = new Connection(neuron1, neuron2);
				neuron1->AddOutConnection(connection);
				neuron2->AddInConnection(connection);
			}
		}
	}
	
	// add new connections between hidden_neurons and hidden_neurons
	for(std::set<Neuron *>::iterator it1 = hidden_neurons.begin(); it1 != hidden_neurons.end(); it1++)
	{
		Neuron *neuron1 = *it1;
		for(std::set<Neuron *>::iterator it2 = hidden_neurons.begin(); it2 != hidden_neurons.end(); it2++)
		{
			Neuron *neuron2 = *it2;
			if(neuron1 != neuron2 && fRand(0,1) < rateHiddenHidden)
			{
				if(neuron1->ContainOutNeuron(neuron2) == false && Reachable(neuron2, neuron1) == false)
				{
					Connection *connection = new Connection(neuron1, neuron2);
					neuron1->AddOutConnection(connection);
					neuron2->AddInConnection(connection);
				}
				else if(neuron2->ContainOutNeuron(neuron1) == false && Reachable(neuron1, neuron2) == false)
				{
					Connection *connection = new Connection(neuron2, neuron1);
					neuron2->AddOutConnection(connection);
					neuron1->AddInConnection(connection);
				}
			}
		}
	}
}

void Network::Eliminate()
{
	for(std::vector<Neuron *>::iterator it1 = input_neurons.begin(); it1 != input_neurons.end(); it1++)
	{
		Neuron *neuron1 = *it1;
		for(std::set<Connection *>::iterator it2 = neuron1->outConnections.begin(); it2 != neuron1->outConnections.end(); it2++)
		{
			Connection *connection = *it2;
			Neuron *neuron2 = connection->outNeuron;
			
			if(fabs(connection->weight) < 1e-8)
			{
				neuron1->outConnections.erase(connection);
				neuron2->inConnections.erase(connection);
				delete connection;
			}
		}
	}
	for(std::set<Neuron *>::iterator it1 = hidden_neurons.begin(); it1 != hidden_neurons.end(); it1++)
	{
		Neuron *neuron1 = *it1;
		for(std::set<Connection *>::iterator it2 = neuron1->outConnections.begin(); it2 != neuron1->outConnections.end(); it2++)
		{
			Connection *connection = *it2;
			Neuron *neuron2 = connection->outNeuron;
			
			if(fabs(connection->weight) < 1e-8)
			{
				neuron1->outConnections.erase(connection);
				neuron2->inConnections.erase(connection);
				delete connection;
			}
		}
	}
	
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->inConnections.size() == 0 && neuron->outConnections.size() == 0)
		{
			hidden_neurons.erase(neuron);
			delete neuron;
		}
	}
}

void Network::Softmax()
{
	double maxValue = -std::numeric_limits<double>::max();
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->value > maxValue)
		{
			maxValue = neuron->value;
		}
	}
	
	double sumExpValue = 0;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->activeValue = neuron->value - maxValue;
		sumExpValue += exp(neuron->activeValue);
	}
	
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->activeValue = exp(neuron->activeValue) / sumExpValue;
	}
}

double Network::CalError()
{
	double error = 0;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		error += neuron->CalError(); // For mean square error
		//if(neuron->trueValue > 0.5) return -log(neuron->activeValue); // For cross-entropy error
	}
	return error;
}

// calculate the number of hidden neurons whose activeValue = 0
int Network::CalZeroCnt()
{
	int zeroCnt = 0;
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(fabs(neuron->activeValue) < eps)
		{
			zeroCnt++;
		}
	}
	return zeroCnt;
}

// calculate the number of hidden neurons in the network
int Network::CalNeuronNum()
{
	return hidden_neurons.size();
}

// calculate the number of connections in the network
int Network::CalConnectionNum()
{
	int sum = 0;
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		sum += neuron->outConnections.size();
	}
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		sum += neuron->outConnections.size();
	}
	return sum;
}

double Network::CalAverageWeight()
{
	if(CalConnectionNum() == 0)
	{
		return 0;
	}
	
	double sum = 0;
	for(std::vector<Neuron *>::iterator it1 = input_neurons.begin(); it1 != input_neurons.end(); it1++)
	{
		Neuron *neuron = *it1;
		for(std::set<Connection *>::iterator it2 = neuron->outConnections.begin(); it2 != neuron->outConnections.end(); it2++)
		{
			sum += fabs((*it2)->weight);
		}
	}
	for(std::set<Neuron *>::iterator it1 = hidden_neurons.begin(); it1 != hidden_neurons.end(); it1++)
	{
		Neuron *neuron = *it1;
		for(std::set<Connection *>::iterator it2 = neuron->outConnections.begin(); it2 != neuron->outConnections.end(); it2++)
		{
			sum += fabs((*it2)->weight);
		}
	}
	return sum / CalConnectionNum();
}

int Network::CalMaxLabel()
{
	double maxValue = -1;
	int maxLabel = -1;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->activeValue > maxValue)
		{
			maxValue = neuron->activeValue;
			maxLabel = neuron->outputTag;
		}
	}
	return maxLabel;
}

double Network::CalCertainty()
{
	double maxValue = -1;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		if(neuron->activeValue > maxValue)
		{
			maxValue = neuron->activeValue;
		}
	}
	return maxValue;
}

bool Network::Reachable(Neuron *s, Neuron *t)
{
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->visited = false;
	}
	
	std::queue<Neuron *> que;
	s->visited = true;
	que.push(s);
	while(!que.empty())
	{
		Neuron *neuron = que.front();
		que.pop();
		if(neuron == t)
		{
			return true;
		}
		for(std::set<Connection *>::iterator it = neuron->outConnections.begin(); it != neuron->outConnections.end(); it++)
		{
			Neuron *outNeuron = (*it)->outNeuron;
			if(outNeuron->visited == false)
			{
				outNeuron->visited = true;
				que.push(outNeuron);
			}
		}
	}
	return false;
}

void Network::AddInputNeuron(Neuron *neuron)
{
	input_neurons.push_back(neuron);
}

void Network::AddHiddenNeuron(Neuron *neuron)
{
	hidden_neurons.insert(neuron);
}

void Network::AddOutputNeuron(Neuron *neuron)
{
	output_neurons.push_back(neuron);
}

void Network::SetInputValue(std::vector<double> input_values)
{
	int N = input_values.size();
	for(int i = 0; i < N; i++)
	{
		input_neurons[i]->value = input_values[i];
	}
}

void Network::SetOutputValue(std::vector<double> output_values)
{
	int N = output_values.size();
	for(int i = 0; i < N; i++)
	{
		output_neurons[i]->trueValue = output_values[i];
	}
}

Network * Network::copy()
{
	int input_N = input_neurons.size();
	int hidden_N = hidden_neurons.size();
	int output_N = output_neurons.size();
	
	Network *new_network = new Network(learning_rate, velocity_decay, regularization_l2, gradientClip); // initialize an empty network
	Neuron *new_input_neurons[input_N];
	Neuron *new_hidden_neurons[hidden_N];
	Neuron *new_output_neurons[output_N];
	
	// set copy tag, used to copy connections
	int cnt = 0;
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->copyTag = cnt;
		new_input_neurons[cnt] = new Neuron(neuron);
		new_network->AddInputNeuron(new_input_neurons[cnt]);
		cnt++;
	}
	
	cnt = 0;
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->copyTag = cnt;
		new_hidden_neurons[cnt] = new Neuron(neuron);
		new_network->AddHiddenNeuron(new_hidden_neurons[cnt]);
		cnt++;
	}
	
	cnt = 0;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->copyTag = cnt;
		new_output_neurons[cnt] = new Neuron(neuron);
		new_network->AddOutputNeuron(new_output_neurons[cnt]);
		cnt++;
	}
	
	// copy connections
	for(std::vector<Neuron *>::iterator it1 = input_neurons.begin(); it1 != input_neurons.end(); it1++)
	{
		Neuron *neuron1 = *it1;
		int copyTag1 = neuron1->copyTag;
		for(std::set<Connection *>::iterator it2 = neuron1->outConnections.begin(); it2 != neuron1->outConnections.end(); it2++)
		{
			Neuron *neuron2 = (*it2)->outNeuron;
			int copyTag2 = neuron2->copyTag;
			double weight = (*it2)->weight;
			
			Connection *connection = new Connection(new_input_neurons[copyTag1], new_hidden_neurons[copyTag2], weight);
			new_input_neurons[copyTag1]->AddOutConnection(connection);
			new_hidden_neurons[copyTag2]->AddInConnection(connection);
		}
	}
	
	for(std::set<Neuron *>::iterator it1 = hidden_neurons.begin(); it1 != hidden_neurons.end(); it1++)
	{
		Neuron *neuron1 = *it1;
		int copyTag1 = neuron1->copyTag;
		for(std::set<Connection *>::iterator it2 = neuron1->outConnections.begin(); it2 != neuron1->outConnections.end(); it2++)
		{
			Neuron *neuron2 = (*it2)->outNeuron;
			int copyTag2 = neuron2->copyTag;
			double weight = (*it2)->weight;
			
			if(neuron2->type == Neuron::hidden)
			{
				Connection *connection = new Connection(new_hidden_neurons[copyTag1], new_hidden_neurons[copyTag2], weight);
				new_hidden_neurons[copyTag1]->AddOutConnection(connection);
				new_hidden_neurons[copyTag2]->AddInConnection(connection);
			}
			else
			{
				Connection *connection = new Connection(new_hidden_neurons[copyTag1], new_output_neurons[copyTag2], weight);
				new_hidden_neurons[copyTag1]->AddOutConnection(connection);
				new_output_neurons[copyTag2]->AddInConnection(connection);
			}
		}
	}
	
	return new_network;
}

void Network::Display()
{
	int cnt = 0;
	for(std::vector<Neuron *>::iterator it = input_neurons.begin(); it != input_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->displayTag = cnt++;
	}
	cnt = 0;
	for(std::set<Neuron *>::iterator it = hidden_neurons.begin(); it != hidden_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->displayTag = cnt++;
	}
	cnt = 0;
	for(std::vector<Neuron *>::iterator it = output_neurons.begin(); it != output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		neuron->displayTag = cnt++;
	}
	
	std::cout << "display network structure : \n";
	for(std::vector<Neuron *>::iterator it1 = input_neurons.begin(); it1 != input_neurons.end(); it1++)
	{
		Neuron *neuron = *it1;
		std::cout << "input " << neuron->displayTag << " : ";
		for(std::set<Connection *>::iterator it2 = neuron->outConnections.begin(); it2 != neuron->outConnections.end(); it2++)
		{
			Connection *connection = *it2;
			std::cout<< "hidden " << connection->outNeuron->displayTag << " weight " << connection->weight << " , ";
		}
		std::cout << "\n";
	}
	for(std::set<Neuron *>::iterator it1 = hidden_neurons.begin(); it1 != hidden_neurons.end(); it1++)
	{
		Neuron *neuron = *it1;
		std::cout << "hidden " << neuron->displayTag << " ( bias " << neuron->bias << " ) : ";
		for(std::set<Connection *>::iterator it2 = neuron->outConnections.begin(); it2 != neuron->outConnections.end(); it2++)
		{
			Connection *connection = *it2;
			if(connection->outNeuron->type == Neuron::hidden)
			{
				std::cout << "hidden " << connection->outNeuron->displayTag << " weight " << connection->weight << " , ";
			}
			else
			{
				std::cout << "out " << connection->outNeuron->displayTag << " weight " << connection->weight << " , ";
			}
		}
		std::cout << "\n";
	}
	for(std::vector<Neuron *>::iterator it1 = output_neurons.begin(); it1 != output_neurons.end(); it1++)
	{
		Neuron *neuron = *it1;
		std::cout << "output " << neuron->displayTag << " ( bias " << neuron->bias << " ) \n";
	}
	std::cout << "\n";
}
