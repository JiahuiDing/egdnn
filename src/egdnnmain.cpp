#include "egdnnmain.h"
#include "neuron.h"
#include "connection.h"
#include "network.h"
#include <iostream>
#include <cstdio>
#include <unistd.h>
#include <cmath>
using namespace EGDNN;

double X_data[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
double Y_data[4] = { 0, 1, 1, 0 };

int main(int argc, char *argv[])
{
	srand(getpid());
	Neuron *neuron1 = new Neuron(Neuron::input, 1);
	Neuron *neuron2 = new Neuron(Neuron::input, 2);
	Neuron *neuron3 = new Neuron(Neuron::hidden, 3);
	Neuron *neuron4 = new Neuron(Neuron::hidden, 4);
	Neuron *neuron5 = new Neuron(Neuron::hidden, 5);
	Neuron *neuron6 = new Neuron(Neuron::hidden, 6);
	Neuron *neuron7 = new Neuron(Neuron::output, 7);
	
	neuron1->AddOutNeuron(neuron3);
	neuron1->AddOutNeuron(neuron4);
	neuron1->AddOutNeuron(neuron5);
	neuron1->AddOutNeuron(neuron6);
	neuron2->AddOutNeuron(neuron3);
	neuron2->AddOutNeuron(neuron4);
	neuron2->AddOutNeuron(neuron5);
	neuron2->AddOutNeuron(neuron6);
	neuron3->AddOutNeuron(neuron7);
	neuron4->AddOutNeuron(neuron7);
	neuron5->AddOutNeuron(neuron7);
	neuron6->AddOutNeuron(neuron7);
	
	neuron3->AddInNeuron(neuron1);
	neuron4->AddInNeuron(neuron1);
	neuron5->AddInNeuron(neuron1);
	neuron6->AddInNeuron(neuron1);
	neuron3->AddInNeuron(neuron2);
	neuron4->AddInNeuron(neuron2);
	neuron5->AddInNeuron(neuron2);
	neuron6->AddInNeuron(neuron2);
	neuron7->AddInNeuron(neuron3);
	neuron7->AddInNeuron(neuron4);
	neuron7->AddInNeuron(neuron5);
	neuron7->AddInNeuron(neuron6);
	
	Network network;
	network.AddNeuron(neuron1);
	network.AddNeuron(neuron2);
	network.AddNeuron(neuron3);
	network.AddNeuron(neuron4);
	network.AddNeuron(neuron5);
	network.AddNeuron(neuron6);
	network.AddNeuron(neuron7);
	
	double lastError = 100000;
	int cnt = 0;
	while(true)
	{
		double error = 0;
		for(int i = 0; i < 4; i++)
		{
			neuron1->value = X_data[i][0];
			neuron2->value = X_data[i][1];
			neuron7->trueValue = Y_data[i];
			
			// network.Display();
			network.ForwardPropagation();
			network.BackPropagation();
			network.UpdateWeight();
			
			error += (neuron7->value - neuron7->trueValue) * (neuron7->value - neuron7->trueValue);
			
			// getchar();
		}
		// getchar();
		std::cout << cnt++ << " " << error / 4 << "\n";
		if(fabs(lastError - error) < 1e-8)
		{
			break;
		}
		lastError = error;
	}
	
	return 0;
}
