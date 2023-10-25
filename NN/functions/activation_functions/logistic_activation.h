#pragma once
#include <cuda_runtime.h>
#include "../../helpers/string_literal.h"
namespace NN {
	namespace Activations {
		

		struct LogisticActivation final {
			static constexpr StringLiteral name = StringLiteral("LogisticActivation");
			/// <summary>
			/// Given an input, apply the activation function
			/// </summary>
			/// <param name="x">input</param>
			/// <returns>logistic function applied to x</returns>
			
			template<typename T>
			__host__ __device__ static T apply(const T x) {
				return 1 / (1 + exp(-x));
			}

			/// <summary>
			/// Given an OUTPUT, compute the derivative with respect to the input
			/// </summary>
			/// <param name="x">The output of the function</param>
			/// <returns>The derivative with respect to the input</returns>
			 
			template<typename T>
			__host__ __device__ static T derivative(const T x) {
				return x * (1 - x);
			}
		};
	}
}