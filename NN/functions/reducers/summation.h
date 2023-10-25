#pragma once
#include <cuda_runtime.h>
#include <concepts>
#include "../../helpers/string_literal.h"
namespace NN {
	namespace Reducers {

		/// <summary>
		/// Sums all arguments supplied to ::apply
		/// Fails if the types are not all equal or the operator+ does not return this type too
		/// 
		/// Computes the partial derivative given the output and the input to differentiate with respect to
		/// </summary>
		struct Summation {
			static constexpr StringLiteral name = "Summation";

			template<typename Type0, typename ... Types>
			static __host__ __device__ inline Type0 apply(Type0 value0, Types... values) {


				static_assert((true & ... & std::same_as<Type0, Types>), "Expected all inputs to be the same types");
				return (value0 + ... + values);
			}

			template<typename Type, typename ... Types>
			static __host__ __device__ inline Type derivative(Type output, Type wrtInput) {
				return 1; // if y = a + b + c + d, consider b. y = b+k where k = a+c+d, dy/db = 1
			}
		};
	}
}