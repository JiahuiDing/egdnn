#include "helper.h"

namespace EGDNN
{
	double eps = 1e-12;
	double learning_rate = 1e-3;
	
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
}
