#pragma once
#include <cuda_runtime.h>

#include "../../helpers/string_literal.h"
namespace NN {
	namespace Activations {
		/// <summary>
		/// Leaky Relu Activation Function
		/// </summary>
		/// <typeparam name="SlopeT">Type of the multiplier for negative values</typeparam>
		/// <typeparam name="slope">value of the multiplier for negative values</typeparam>
		template<typename SlopeT = double, SlopeT slope = 0.01>
		struct LeakyReluActivation {
			static constexpr StringLiteral name = StringLiteral("LeakyReluActivation");

			template<typename T>
			static __host__ __device__ T apply(T x) {
				return static_cast<T>((x > 0) ? x : slope * x);
			}
			template<typename T>
			static __host__ __device__ T derivative(T x) {
				return static_cast<T>((x > 0) ? 1 : slope);
			}
		};

	}
}