#include "connection.h"
using namespace EGDNN;

Connection::Connection(Neuron *neuron) : neuron(neuron)
{
	weight = fRand(-1, 1);
}
