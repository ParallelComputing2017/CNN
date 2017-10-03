/*
 * Net2.h
 *
 *  Created on: 19/08/2017
 *      Author: juan
 */

#ifndef NEURALNETWORK_NEURON_HPP_
#define NEURALNETWORK_NEURON_HPP_

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection {
	double weight;
	double deltaWeight;
};

// ****************** class Neuron ******************

class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) {
		m_outputVal = val;
	}
	double getOutputVal(void) const;
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVals);
	void calcHiddenGradients(const Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);
private:
	static double eta; // [0.0...1.0] overall net training rate
	static double alpha; // [0.0...n] multiplier of last weight change [momentum]
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	// randomWeight: 0 - 1
	static double randomWeight(void) {
		return rand() / double(RAND_MAX);
	}
	double sumDOW(const Layer &nextLayer) const;
	double m_outputVal;
	std::vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
};




#endif /* NEURALNETWORK_NET_HPP_ */
