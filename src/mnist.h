#pragma once

#include "types.h"

using namespace std;

#pragma pack(push, 1)
struct RGB {
	uint8_t r, g, b;
};
#pragma pack(pop)

uint8_t* read_file(const char* szFile) {
	ifstream file(szFile, ios::binary | ios::ate);
	streamsize size = file.tellg();
	file.seekg(0, ios::beg);

	if (size == -1)
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read((char*) buffer, size);
	return buffer;
}

vector<case_t> read_cases(string imagesFile, string labelsFile) {
	vector<case_t> cases;

	uint8_t* train_image = read_file(imagesFile.c_str());
	uint8_t* train_labels = read_file(labelsFile.c_str());

	uint32_t case_count = byteswap_uint32(*(uint32_t*) (train_image + 4));

	for (int i = 0; i < case_count; i++) {
		case_t c { tensor_t<float>(28, 28, 1), tensor_t<float>(10, 1, 1) };

		uint8_t* img = train_image + 16 + i * (28 * 28);
		uint8_t* label = train_labels + 8 + i;

		for (int x = 0; x < 28; x++)
			for (int y = 0; y < 28; y++)
				c.data(x, y, 0) = img[x + y * 28] / 255.f;

		for (int b = 0; b < 10; b++)
			c.out(b, 0, 0) = *label == b ? 1.0f : 0.0f;

		cases.push_back(c);
	}
	delete[] train_image;
	delete[] train_labels;

	return cases;
}

vector<case_t> read_training_cases() {

	string trainImages = "data/mnist/train-images.idx3-ubyte";
	string trainLabels = "data/mnist/train-labels.idx1-ubyte";
	vector<case_t> cases = read_cases(trainImages, trainLabels);

	return cases;
}

vector<case_t> readTestCases() {

	string images = "data/mnist/t10k-images.idx3-ubyte";
	string labels = "data/mnist/t10k-labels.idx1-ubyte";
	vector<case_t> cases = read_cases(images, labels);

	return cases;
}

float fullTest(vector<layer_t*>& master) {
	vector<case_t> cases = readTestCases();

	Logger::info("Full test. Test cases: %lu \n", cases.size());

	case_t &sample = cases.at(1);

	NeuralNetwork neuralNetwork (master);

	neuralNetwork.test(sample.data);

	tensor_t<float> out = master.back()->out - sample.out;

	Logger::debug("Sample: ");
	// TODO
	//print_tensor(sample.out);

	Logger::debug("Net: ");
	// TODO
	//print_tensor(master.back()->out);

	float accumulativeError = 0.0;

	int classes = sample.out.getSize().x * sample.out.getSize().y
			* sample.out.getSize().z;

	for (case_t sample : cases) {

		neuralNetwork.test(sample.data);

		for (int i = 0; i < classes; i++) {
			float expected = sample.out.get(i, 0, 0);
			if (expected == 1.0) {
				accumulativeError += expected - master.back()->out.get(i, 0, 0);
				break;
			}

		}
	}
	float error = accumulativeError / cases.size();

	Logger::info("Full test. Error: %f \n", error);

	return error;
}

int singleTest(vector<layer_t*> master) {
	uint8_t * data = read_file("data/test.ppm");

	Logger::info("Single test");

	NeuralNetwork neuralNetwork (master);

	int digit = -1;

	if (data) {
		uint8_t * usable = data;

		while (*(uint32_t*) usable != 0x0A353532)
			usable++;

		RGB * rgb = (RGB*) usable;

		tensor_t<float> image(28, 28, 1);
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) {
				RGB rgb_ij = rgb[i * 28 + j];
				image(j, i, 0) = (((float) rgb_ij.r + rgb_ij.g + rgb_ij.b)
						/ (3.0f * 255.f));
			}
		}

		neuralNetwork.test(image);
		tensor_t<float>& out = master.back()->out;

		float maxProbability = 0.0;

		for (int i = 0; i < 10; i++) {
			float probability = out(i, 0, 0) * 100.0f;
			if (probability > maxProbability) {
				digit = i;
				maxProbability = probability;
			}
			printf("[%i] %.3f    ", i, probability);
		}
		printf("\n");

		delete[] data;
	}

	return digit;
}
