#pragma once
#include <cuda_runtime.h>

namespace NN {
	namespace ErrorMeasures {

		struct SquaredError final {
			template<typename T>
			__host__ __device__ static T apply(const T output, const T expected)  {
				return static_cast<T>(pow((output - expected), 2));
			}
			template<typename T>
			__host__ __device__ static T derivative(const T output, const T expected) {
				return 2 * (output - expected);
			}
			template<typename T>
			__host__ __device__ static T average(const T* outputs, const T* expected, const unsigned int size) {
				T sum = 0;
				for (unsigned int i = 0; i < size; i++) {
					sum += apply(outputs[i], expected[i]) / size;
				}
				return sum;
			}
		};
	}
}