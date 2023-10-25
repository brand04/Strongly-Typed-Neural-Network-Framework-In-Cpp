#pragma once
#pragma once
#include <cuda_runtime.h>

namespace NN {
	namespace WeightModifiers {

		/*
		Subtracts a portion of delta from the weight
		*/
		template<typename T, const T Threshold = (T)(0.5), const T LearningRate = 0.000001 >
		struct ClampedLogistic final {
			__host__ __device__ static void adjust(T& weight, const T delta) {
				T trueDelta = LearningRate * (1 / (1 - exp(-delta)));
				if (trueDelta > Threshold) weight -= Threshold;
				else if (trueDelta < -Threshold) weight += Threshold;
				else weight -= trueDelta;
			}
		};
	}


}