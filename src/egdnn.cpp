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

void Egdnn::fit(std::vector<std::vector<double>> trainingSet, std::vector<std::vector<double>> trainingLabels, int training_N, int maxIter, int batchSize, int evolutionTime)
{
	struct timeval start, end;
	gettimeofday(&start, NULL);
	for(int iterCnt = 0; iterCnt < maxIter && kbhit() == false; iterCnt++)
	{
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
		
		for(int evolutionCnt = 0; evolutionCnt < evolutionTime; evolutionCnt++)
		{
			for(int batchCnt = 0; batchCnt < batchSize; batchCnt++)
			{
				int data_i = rand() % training_N;
				for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
				{
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
				network[networkCnt]->UpdateWeight();
			}
		}
		
		double minError = 1e9;
		int bestNetwork = -1;
		std::cout << "iter " << iterCnt << " performance : \n";
		for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
		{
			std::cout << "network " << networkCnt << " : ";
			std::cout << std::setprecision(10) << "error " << error[networkCnt] / (evolutionTime * batchSize) << " , ";
			std::cout << "accuracy " << (double)rightCnt[networkCnt] / (evolutionTime * batchSize) << " , ";
			std::cout << "neuronNum " << network[networkCnt]->CalNeuronNum() << " , ";
			std::cout << "connectionNum " << network[networkCnt]->CalConnectionNum() << " , ";
			std::cout << "learning_rate " << network[networkCnt]->learning_rate << " , ";
			std::cout << "velocity_decay " << network[networkCnt]->velocity_decay << " , ";
			std::cout << "zeroRate " << (double)zeroCnt[networkCnt] / (evolutionTime * batchSize * network[networkCnt]->CalNeuronNum()) << " , ";
			std::cout << "certainty " << certainty[networkCnt] / (evolutionTime * batchSize) << " , ";
			std::cout << "averageWeight " << network[networkCnt]->CalAverageWeight() << "\n";
			
			if(error[networkCnt] < minError)
			{
				minError = error[networkCnt];
				bestNetwork = networkCnt;
			}
		}
		gettimeofday(&end, NULL);
		double timeuse = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
		std::cout << "time : " << timeuse / 1000000 << " s\n\n";
		gettimeofday(&start, NULL);
		
		// kill all networks except the best
		for(int networkCnt = 0; networkCnt < populationSize; networkCnt++)
		{
			if(networkCnt != bestNetwork)
			{
				delete network[networkCnt];
			}
		}
		
		// reproduce
		network[0] = network[bestNetwork];
		network[0]->Eliminate();
		for(int networkCnt = 1; networkCnt < populationSize; networkCnt++)
		{
			network[networkCnt] = network[0]->copy();
			network[networkCnt]->Mutate();
		}
		//network[0]->Display();
	}
}

void Egdnn::test(std::vector<std::vector<double>> testSet, std::vector<std::vector<double>> testLabels, int test_N)
{
	double error = 0;
	int rightCnt = 0;
	for(int data_i = 0; data_i < test_N; data_i++)
	{
		network[0]->SetInputValue(testSet[data_i]);
		network[0]->SetOutputValue(testLabels[data_i]);
		
		network[0]->ForwardPropagation();
		
		error += network[0]->CalError();
		if(testLabels[data_i][network[0]->CalMaxLabel()] > 0.5)
		{
			rightCnt++;
		}
	}
	std::cout << "test error : " << error / test_N << "\n";
	std::cout << "test accuracy : " << (double) rightCnt / test_N << "\n\n";
}
