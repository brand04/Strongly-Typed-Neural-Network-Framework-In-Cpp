#pragma once
#include <cuda_runtime.h>
#include <concepts>
#include "../../helpers/string_literal.h"
namespace NN {
	namespace Reducers {

		/// <summary>
		/// Multiplies all arguments supplied to ::apply
		/// Fails if the types are not all equal or the operator* does not return this type too
		/// 
		/// Computes the partial derivative given the output and the input to differentiate with respect to
		/// </summary>
		struct DotProduct {
			static constexpr StringLiteral name = "DotProduct";

			template<typename Type0, typename ... Types>
			static __host__ __device__ inline Type0 apply(Type0 value0, Types... values) {

				
				static_assert((true & ... & std::same_as<Type0, Types>), "Expected all inputs to be the same types");
				return (value0 * ... * values);
			}

			template<typename Type, typename ... Types>
			static __host__ __device__ inline Type derivative(Type output, Type wrtInput) {
				return (wrtInput == static_cast<Type>(0)) ? (0.0001) : (output / wrtInput); //The partial derivative is computable given output y and input x for all x != 0, however cannot be computed this way if x=0, however what we can do is cause a nudge that will dislodge from x=0 to a computable gradient
			}
		};
	}
}