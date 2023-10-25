#pragma once
#include <cuda_runtime.h>

namespace NN {
	namespace ErrorMeasures {
        
		struct HalvedSquaredError final {
            template<typename T>
            __host__ __device__ static T apply(const T output, const T expected) {
                return (pow((output - expected), 2) / 2);
            }
            template<typename T>
            __host__ __device__ static T derivative(const T output, const T expected)  {
                return output - expected;
            } 
            template<typename T>
            __host__ __device__ static T average(const T* outputs, const T* expected, const unsigned int size) {
                T sum = 0; 
                for (unsigned int i = 0; i<size; i++) {
                    sum += apply(outputs[i], expected[i]) / size;
                }
                return sum;
            }
        };
	}
}