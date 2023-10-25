#pragma once
#include <cuda_runtime.h>
#include "../../helpers/string_literal.h"
namespace NN {
	namespace Activations {

		/// <summary>
		/// A function that does nothing to the input
		/// </summary>
		struct IdentityActivation final {

			static constexpr StringLiteral name = StringLiteral("IdentityActivation");

			template<typename T>
			static __host__ __device__ T apply(const T x) {
				return x;
			}
			template<typename T>
			static __host__ __device__ T derivative(const T x) {
				return 1;
			}
		};
	}
}