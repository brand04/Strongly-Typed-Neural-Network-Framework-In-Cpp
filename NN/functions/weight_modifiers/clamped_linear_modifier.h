#pragma once
#include <cuda_runtime.h>

namespace NN {
	namespace WeightModifiers {

		/*
		Subtracts a portion of delta from the weight, up to a certain amount
		*/
		template<typename T, const T threshold = (T)0.0001, const T LearningRate = 0.0000001>
		struct ClampedLinear final {
			__host__ __device__ static void adjust(T& weight, const T delta) {
				T trueDelta = LearningRate * delta;
				if (trueDelta > threshold) weight-=threshold;
				else if (trueDelta < -threshold) weight+=threshold;
				else weight -= trueDelta;
			}
		};
	}
}