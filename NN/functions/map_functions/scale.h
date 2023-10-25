#pragma once
#include "../../dtypes/concept.h"
#include <cuda_runtime.h>
namespace NN {
	namespace MapFunctions {

		template<Dtypes::Dtype Dtype, auto scalar> requires (std::same_as<decltype(scalar),typename Dtype::Type>)
		struct Scale {
			static __host__ __device__ void apply(typename Dtype::Type& value) {
				value*=scalar;
			}
		};
	}

}