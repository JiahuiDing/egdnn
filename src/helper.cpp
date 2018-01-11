#include "helper.h"

namespace EGDNN
{
	double eps = 1e-12;
	
	double fRand(double fMin, double fMax)
	{
		double f = (double)rand() / RAND_MAX;
		return fMin + f * (fMax - fMin);
	}
	
	uchar** read_mnist_images(std::string full_path, int& number_of_images, int& image_size) {

		/*
		    [offset] [type]          [value]          [description]
		    0000     32 bit integer  0x00000803(2051) magic number
		    0004     32 bit integer  10000 or 60000   number of images
		    0008     32 bit integer  28               number of rows
		    0012     32 bit integer  28               number of columns
		    0016     unsigned byte   ??               pixel
		    0017     unsigned byte   ??               pixel
		    ........
		    xxxx     unsigned byte   ??               pixel
		    Pixels are organized row-wise. Pixel values are 0 to 255.
		    0 means background (white), 255 means foreground (black).
		*/

		// Users of Intel processors and other low-endian machines must flip the bytes of the header.
		auto reverseInt = [](int i) {
		    unsigned char c1, c2, c3, c4;
		    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
		};

		std::ifstream file(full_path);

		if(file.is_open()) {
		    int magic_number = 0, n_rows = 0, n_cols = 0;

		    file.read((char *)&magic_number, sizeof(magic_number));
		    magic_number = reverseInt(magic_number);

		    if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

		    file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		    file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		    file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		    image_size = n_rows * n_cols;

		    uchar** _dataset = new uchar*[number_of_images];
		    for(int i = 0; i < number_of_images; i++) {
		        _dataset[i] = new uchar[image_size];
		        file.read((char *)_dataset[i], image_size);
		    }

		    // _dataset[number_of_images][image_size]
		    return _dataset;
		} else {
		    throw std::runtime_error("Cannot open file `" + full_path + "`!");
		}
	}

	uchar* read_mnist_labels(std::string full_path, int number_of_images) {

		/*
		   [offset] [type]          [value]          [description]
		   0000     32 bit integer  0x00000801(2049) magic number (MSB first)
		   0004     32 bit integer  10000 or 60000   number of items
		   0008     unsigned byte   ??               label
		   0009     unsigned byte   ??               label
		   ........
		   xxxx     unsigned byte   ??               label
		   The labels values are 0 to 9.
		 */

		// Users of Intel processors and other low-endian machines must flip the bytes of the header.
		auto reverseInt = [](int i) {
		    unsigned char c1, c2, c3, c4;
		    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
		};

		std::ifstream file(full_path, std::fstream::in);

		if(file.is_open()) {
		    int magic_number = 0, n_images = 0;

		    file.read((char *)&magic_number, sizeof(magic_number));
		    magic_number = reverseInt(magic_number);

		    if(magic_number != 2049) throw std::runtime_error("Invalid MNIST image file!");

		    file.read((char *)&n_images, sizeof(n_images)), n_images = reverseInt(n_images);

		    if(number_of_images != n_images) throw std::runtime_error("Labels don't correspond to the previous dataset!");

		    uchar* _labels = new uchar[number_of_images]();

		    uchar uch;
		    int i = 0;

		    while (file >> std::noskipws >> uch) {
		        _labels[i++] = uch;
		    }

		    // _labels[number_of_images]
		    return _labels;
		} else {
		    throw std::runtime_error("Cannot open file `" + full_path + "`!");
		}
	}
	
	void read_mnist(std::vector<std::vector<double>> &trainingSet, std::vector<std::vector<double>> &trainingLabels, int &training_N, 
					std::vector<std::vector<double>> &testSet, std::vector<std::vector<double>> &testLabels, int &test_N, 
					int &input_N, int &output_N)
	{
		std::cout << "Begin loading mnist dataset!\n";
		output_N = 10;
		uchar **trainingSetMnist = read_mnist_images("mnist/train-images.idx3-ubyte", training_N, input_N);
		uchar *trainingLabelsMnist = read_mnist_labels("mnist/train-labels.idx1-ubyte", training_N);
		uchar **testSetMnist = read_mnist_images("mnist/t10k-images.idx3-ubyte", test_N, input_N);
		uchar *testLabelsMnist = read_mnist_labels("mnist/t10k-labels.idx1-ubyte", test_N);
		
		trainingSet.resize(training_N);
		for(int i = 0; i < training_N; i++)
		{
			trainingSet[i].resize(input_N);
		}
	
		trainingLabels.resize(training_N);
		for(int i = 0; i < training_N; i++)
		{
			trainingLabels[i].resize(output_N);
		}
	
		testSet.resize(test_N);
		for(int i = 0; i < test_N; i++)
		{
			testSet[i].resize(input_N);
		}
	
		testLabels.resize(test_N);
		for(int i = 0; i < test_N; i++)
		{
			testLabels[i].resize(output_N);
		}
	
		for(int i = 0; i < training_N; i++)
		{
			for(int j = 0; j < input_N; j++)
			{
				trainingSet[i][j] = (double)trainingSetMnist[i][j] / 255;
			}
		}
	
		for(int i = 0; i < training_N; i++)
		{
			for(int j = 0; j < output_N; j++)
			{
				trainingLabels[i][j] = 0;
			}
			trainingLabels[i][(int)trainingLabelsMnist[i]] = 1;
		}
	
		for(int i = 0; i < test_N; i++)
		{
			for(int j = 0; j < input_N; j++)
			{
				testSet[i][j] = (double)testSetMnist[i][j] / 255;
			}
		}
	
		for(int i = 0; i < test_N; i++)
		{
			for(int j = 0; j < output_N; j++)
			{
				testLabels[i][j] = 0;
			}
			testLabels[i][(int)testLabelsMnist[i]] = 1;
		}
		
		std::cout << "Finish loading mnist dataset!\n";
	}
}
