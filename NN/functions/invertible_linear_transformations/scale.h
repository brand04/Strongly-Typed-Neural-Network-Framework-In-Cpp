#pragma once
#include <cuda_runtime.h>
namespace NN {
	namespace Functions {
		namespace InvertibleLinearTransformations {

			/// <summary>
			/// Scales the input by a constant factor
			/// </summary>
			/// <typeparam name="T"></typeparam>
			/// <typeparam name="scalar"></typeparam>
			template<typename T, T scalar>
			struct Scale {

				static __host__ __device__ void apply(T& value) {
					value *= scalar;
				}

				static __host__ __device__ void inverse(T& value) {
					value /= scalar;
				}
			};
		}
	}
}