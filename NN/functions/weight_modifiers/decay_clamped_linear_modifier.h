#pragma once
#include <cuda_runtime.h>

namespace NN {
	namespace WeightModifiers {

		/*
		Subtracts a portion of delta from the weight, preventing crossing weight = 0 if the delta is bigger than threshold and instead decaying the value by an amount
		*/
		template<typename T, const T threshold = (T)0.0001, const T decayFactor = 0.5, const T LearningRate = 0.0000001>
		struct DecayClampedLinear final {
			__host__ __device__ static void adjust(T& weight, const T delta) {
				static_assert(threshold > 0, "Threshold should be a positive value");
				T trueDelta = LearningRate * delta;
				if (trueDelta > threshold) {
					weight *= decayFactor;
				}
				else if (trueDelta < -threshold) {
					weight *= decayFactor;
				}
				else {
					weight -= trueDelta;
				}
			}
		};
	}
}