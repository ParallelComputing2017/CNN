#pragma once
#include <cassert>
#include <vector>
#include <string.h>

#include "point_t.h"
#include "Common/logger.h"

template<typename T>
struct tensor_t {
	T * data;

	tdsize size;

	tensor_t(int _x, int _y, int _z) {
		data = new T[_x * _y * _z];
		size.x = _x;
		size.y = _y;
		size.z = _z;
	}

	tensor_t(tdsize size) :
			size(size) {

		data = new T[size.x * size.y * size.z];
	}

	tensor_t(const tensor_t& other) {
		data = new T[other.size.x * other.size.y * other.size.z];
		memcpy(this->data, other.data,
				other.size.x * other.size.y * other.size.z * sizeof(T));
		this->size = other.size;
	}

	tensor_t<T> operator+(tensor_t<T>& other) {
		tensor_t<T> clone(*this);
		for (int i = 0; i < other.size.x * other.size.y * other.size.z; i++) {
			clone.data[i] += other.data[i];
		}

		return clone;
	}

	tensor_t<T> operator-(tensor_t<T>& other) {

		assert(
				size.x == other.size.x && size.y == other.size.y
						&& size.z == other.size.z);

		tensor_t<T> clone(*this);
		for (int i = 0; i < other.size.x * other.size.y * other.size.z; i++)
			clone.data[i] -= other.data[i];
		return clone;
	}

	tensor_t<T> operator/(float divisor) {
		tensor_t<T> clone(*this);
		for (int i = 0; i < clone.size.x * clone.size.y * clone.size.z; i++) {
			clone.data[i] = clone.data[i] / divisor;
		}
		return clone;
	}

	T& operator()(int _x, int _y, int _z) {
		return this->get(_x, _y, _z);
	}

	T& operator==(const tensor_t<T>& other) {
		assert(
				size.x == other.size.x && size.y == other.size.y
						&& size.z == other.size.z);

		for (int i = 0; i < size.x * size.y * size.z; i++) {
			if (data[i] != other.data[i]) {
				return false;
			}
		}

		return true;

	}

	T& get(int _x, int _y, int _z) {
		assert(_x >= 0 && _y >= 0 && _z >= 0);
		assert(_x < size.x && _y < size.y && _z < size.z);

		return data[_z * (size.x * size.y) + _y * (size.x) + _x];
	}

	void copy_from(std::vector<std::vector<std::vector<T>>>data )
	{
		int z = data.size();
		int y = data[0].size();
		int x = data[0][0].size();

		for ( int i = 0; i < x; i++ )
		for ( int j = 0; j < y; j++ )
		for ( int k = 0; k < z; k++ )
		get( i, j, k ) = data[k][j][i];
	}

	~tensor_t()
	{
		delete[] data;
	}

	tdsize getSize() {
		return size;
	}

};

static void print_tensor(tensor_t<float>& data) {
	int mx = data.getSize().x;
	int my = data.getSize().y;
	int mz = data.getSize().z;

	printf("tensor size [ %i, %i, %i ] \n", mx, my, mz);

	for (int z = 0; z < mz; z++) {
		printf("[Dim%d]\n", z);
		for (int y = 0; y < my; y++) {
			for (int x = 0; x < mx; x++) {
				printf("%.2f \t", (float) data.get(x, y, z));
			}
			printf("\n");
		}
	}
}

static tensor_t<float> to_tensor(std::vector<std::vector<std::vector<float>>>data )
{
	int z = data.size();
	int y = data[0].size();
	int x = data[0][0].size();

	tensor_t<float> t( x, y, z );

	for ( int i = 0; i < x; i++ )
	for ( int j = 0; j < y; j++ )
	for ( int k = 0; k < z; k++ )
	t( i, j, k ) = data[k][j][i];
	return t;
}
