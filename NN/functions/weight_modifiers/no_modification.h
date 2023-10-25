#pragma once
#pragma once
#include <cuda_runtime.h>

namespace NN {
	namespace WeightModifiers {

		/*
		Subtracts a portion of delta from the weight, up to a certain amount
		*/
		template<typename T>
		struct NoModification final {
			__host__ __device__ static void adjust(T& weight, const T delta) {}
		};
	}
}