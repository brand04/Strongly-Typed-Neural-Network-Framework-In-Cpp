#pragma once
#include <cuda_runtime.h>

namespace NN {
	namespace WeightModifiers {

		/*
		Subtracts a portion of delta from the weight, scaled by the learningRate, adding offset * sign(delta), clamping by threshold
		*/
		template<typename T, const T threshold = (T)0.0001, const T offset = 0.1, const T LearningRate = 0.0000001>
		struct ClampedOffset final {
			__host__ __device__ static void adjust(T& weight, const T delta) {
				T trueDelta = (LearningRate * delta) + (delta>0 ? offset : -offset);
				if (trueDelta > threshold) weight -= threshold;
				else if (trueDelta < -threshold) weight += threshold;
				else weight -= trueDelta;
			}
		};
	}
}