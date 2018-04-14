#include "egdnn.h"
using namespace EGDNN;

Egdnn::Egdnn(int input_N, int output_N, int populationSize, double learning_rate, double velocity_decay, double regularization_l2, double gradientClip) : 
				input_N(input_N), output_N(output_N), populationSize(populationSize),
				learning_rate(learning_rate), velocity_decay(velocity_decay), regularization_l2(regularization_l2), gradientClip(gradientClip)
				
{
	srand(getpid());
	
	network.resize(populationSize);
	for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
	{
		network[networkCnt] = new Network(learning_rate, velocity_decay, regularization_l2, gradientClip);
		for(int i = 0; i < input_N; i++)
		{
			network[networkCnt]->AddInputNeuron(new Neuron(-1, Neuron::input));
		}
		for(int i = 0; i < output_N; i++)
		{
			network[networkCnt]->AddOutputNeuron(new Neuron(i, Neuron::output));
		}
		network[networkCnt]->Mutate();
	}
}

// netId = -1 means train all networks
void Egdnn::fit(int netId, std::vector<std::vector<double>> trainingSet, std::vector<std::vector<double>> trainingLabels, int iterNum, int batchSize)
{
	int training_N = trainingSet.size();
	struct timeval start, end;
	gettimeofday(&start, NULL);
	
	double error[populationSize];
	int rightCnt[populationSize];
	int zeroCnt[populationSize];
	double certainty[populationSize];
	for(int i = 0; i < populationSize; i++)
	{
		error[i] = 0;
		rightCnt[i] = 0;
		zeroCnt[i] = 0;
		certainty[i] = 0;
	}
	
	// train
	for(int iterCnt = 0; iterCnt < iterNum; iterCnt++)
	{
		for(int batchCnt = 0; batchCnt < batchSize; batchCnt++)
		{
			int data_i = rand() % training_N;
			for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
			{
				if(netId != -1 && netId != networkCnt)
					continue;
					
				network[networkCnt]->SetInputValue(trainingSet[data_i]);
				network[networkCnt]->SetOutputValue(trainingLabels[data_i]);
				network[networkCnt]->ForwardPropagation();
				network[networkCnt]->BackPropagation();
				
				error[networkCnt] += network[networkCnt]->CalError();
				if(trainingLabels[data_i][network[networkCnt]->CalMaxLabel()] > 0.5)
				{
					rightCnt[networkCnt]++;
				}
				zeroCnt[networkCnt] += network[networkCnt]->CalZeroCnt();
				certainty[networkCnt] += network[networkCnt]->CalCertainty();
			}
		}
		
		for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
		{
			if(netId != -1 && netId != networkCnt)
				continue;
				
			network[networkCnt]->UpdateWeight();
		}
	}
	
	/*
	for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
	{
		std::cout << "net " << networkCnt << " : ";
		std::cout << std::setprecision(6) << "error " << error[networkCnt] / (iterNum * batchSize) << " , ";
		std::cout << "accuracy " << (double)rightCnt[networkCnt] / (iterNum * batchSize) << " , ";
		std::cout << "neuronNum " << network[networkCnt]->CalNeuronNum() << " , ";
		std::cout << "connectionNum " << network[networkCnt]->CalConnectionNum() << " , ";
		std::cout << "learning_rate " << network[networkCnt]->learning_rate << " , ";
		std::cout << "velocity_decay " << network[networkCnt]->velocity_decay << " , ";
		std::cout << "zeroRate " << (double)zeroCnt[networkCnt] / (iterNum * batchSize * network[networkCnt]->CalNeuronNum()) << " , ";
		std::cout << "certainty " << certainty[networkCnt] / (iterNum * batchSize) << " , ";
		std::cout << "averageWeight " << network[networkCnt]->CalAverageWeight() << " , ";
		std::cout << "\n";
	}
	*/
	
	gettimeofday(&end, NULL);
	double timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
	//std::cout << "time : " << timeuse / 1000000 << " s\n\n";
}

std::vector<double> Egdnn::predict(int netId, std::vector<double> data)
{
	network[netId]->SetInputValue(data);
	network[netId]->ForwardPropagation();
	
	std::vector<double> result;
	for(std::vector<Neuron *>::iterator it = network[netId]->output_neurons.begin(); it != network[netId]->output_neurons.end(); it++)
	{
		Neuron *neuron = *it;
		result.push_back(neuron->activeValue);
	}
	
	return result;
}

double Egdnn::test(int netId, std::vector<std::vector<double>> testSet, std::vector<std::vector<double>> testLabels)
{
	int test_N = testSet.size();
	double error = 0;
	double accuracy = 0;
	for(int data_i = 0; data_i < test_N; data_i++)
	{
		network[netId]->SetInputValue(testSet[data_i]);
		network[netId]->SetOutputValue(testLabels[data_i]);
		
		network[netId]->ForwardPropagation();
		
		error += network[netId]->CalError();
		if(testLabels[data_i][network[netId]->CalMaxLabel()] > 0.5)
		{
			accuracy++;
		}
	}
	error /= test_N;
	accuracy /= test_N;
	std::cout << "net " << netId << " : ";
	std::cout << "test error " << error << " , ";
	std::cout << "test accuracy " << accuracy << " , ";
	std::cout << "\n";
	return accuracy;
}

void Egdnn::evolution(int bestNetId)
{
	// kill all networks except the best
	for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
	{
		if(networkCnt != bestNetId)
		{
			delete network[networkCnt];
		}
	}
	
	// reproduce
	network[0] = network[bestNetId];
	network[0]->Eliminate();
	for(int networkCnt = 1; networkCnt < populationSize; networkCnt++)
	{
		network[networkCnt] = network[0]->copy();
		network[networkCnt]->Mutate();
	}
	//network[0]->Display();
}

void Egdnn::display()
{
	for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
	{
		std::cout << "net " << networkCnt << " : ";
		std::cout << "neuronNum " << network[networkCnt]->CalNeuronNum() << " , ";
		std::cout << "connectionNum " << network[networkCnt]->CalConnectionNum() << " , ";
		//std::cout << "learning_rate " << network[networkCnt]->learning_rate << " , ";
		//std::cout << "velocity_decay " << network[networkCnt]->velocity_decay << " , ";
		std::cout << "averageWeight " << network[networkCnt]->CalAverageWeight() << " , ";
		std::cout << "\n";
	}
}
