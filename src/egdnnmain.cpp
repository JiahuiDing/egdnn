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
	Neuron *neuron1 = new Neuron(1, Neuron::input);
	Neuron *neuron2 = new Neuron(2, Neuron::input);
	Neuron *neuron3 = new Neuron(3, Neuron::hidden);
	Neuron *neuron4 = new Neuron(4, Neuron::hidden);
	Neuron *neuron5 = new Neuron(5, Neuron::hidden);
	Neuron *neuron6 = new Neuron(6, Neuron::hidden);
	Neuron *neuron7 = new Neuron(7, Neuron::output);
	
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
	
	double lastError = 1e9;
	int cnt = 0;
	while(true)
	{
		std::cout << "iteration : " << cnt++ << "\n";
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
			
			std::cout << "value : " << neuron7->value << " , " << "trueValue : " << neuron7->trueValue << "\n";
			error += network.CalError();
		}
		
		std::cout << " " << error / 4 << "\n";
		if(fabs(lastError - error) < 1e-12)
		{
			break;
		}
		lastError = error;
		
		// getchar();
	}
	
	return 0;
}
