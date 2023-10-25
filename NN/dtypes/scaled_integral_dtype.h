#pragma once
#include "./signed_integral_dtype.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
namespace NN {

	namespace Dtypes {

		//Same as AsDtype<T> but scales the initialization values by scale (useful for larger networks)
		template<typename T, T scale>
		struct AsDtypeScaled : AsDtype<T> {
			static inline __host__ __device__ T  init(unsigned int seed) {
				return AsDtype<T>::init(seed) * scale;
			}

		};
	}
}