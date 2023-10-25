#pragma once
#include <cuda_runtime.h>

namespace NN {
	namespace WeightModifiers {

		/*
		Subtracts a portion of delta from the weight
		*/
		template<typename T, const T LearningRate = 0.000001>
		struct Linear final {
			__host__ __device__ static void adjust(T& weight, const T delta) {
				weight -= (LearningRate * delta);
			}
		};
	}


}