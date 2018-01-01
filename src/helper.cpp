#include "helper.h"

namespace EGDNN
{
	double learning_rate = 0.01;
	
	double fRand(double fMin, double fMax)
	{
		double f = (double)rand() / RAND_MAX;
		return fMin + f * (fMax - fMin);
	}
}
