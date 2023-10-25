#pragma once
#pragma once
#include "./scaled_integral_dtype.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
namespace NN {
	namespace Dtypes {

		//Same as AsDtypeScaled<T> but with an offset added on - allowing for centering values near the offset
		template<typename T, T scale, T offset>
		struct AsDtypeScaledWithOffset : AsDtypeScaled<T,scale> {
			static inline __host__ __device__ T  init(unsigned int seed) {
				return AsDtypeScaled<T,scale>::init(seed) + offset;
			}
		};
	}
}