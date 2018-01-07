#ifndef _HELPER_H_
#define _HELPER_H_

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>

namespace EGDNN
{
	typedef unsigned char uchar;
	
	extern double eps;
	extern double learning_rate;
	
	double fRand(double fMin, double fMax);
	
	/* 
		usage :
		int number_of_images;
		int image_size;
		
		// Read Image
		uchar **dataset = read_mnist_images("t10k-images.idx3-ubyte", number_of_images, image_size);
		// Read Labels
		uchar *labels = read_mnist_labels("t10k-labels.idx1-ubyte", number_of_images);
	*/
	uchar** read_mnist_images(std::string full_path, int& number_of_images, int& image_size);
	uchar* read_mnist_labels(std::string full_path, int number_of_images);
	void read_mnist(std::vector<std::vector<double>> &trainingSet, std::vector<std::vector<double>> &trainingLabels, int &training_N, 
					std::vector<std::vector<double>> &testSet, std::vector<std::vector<double>> &testLabels, int &test_N, 
					int &input_N, int &output_N);
}

#endif
