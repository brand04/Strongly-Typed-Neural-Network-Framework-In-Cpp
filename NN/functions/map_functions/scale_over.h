#pragma once
#pragma once
#include "../../dtypes/concept.h"
#include "../../asserts/false_assert.h"
#include <cuda_runtime.h>
namespace NN {
	namespace MapFunctions {

		//scales the proportion of the input that is over threshold, by a factor of scalar
		template<Dtypes::Dtype Dtype, auto threshold, auto scalar> requires (std::same_as<decltype(scalar), typename Dtype::Type>&& std::same_as<decltype(threshold), typename Dtype::Type>)
			struct ScaleOver {
			static __host__ __device__ void apply(typename Dtype::Type& value) {
				if constexpr (std::is_signed_v<typename Dtype::Type>) { //signed value
					if (value > threshold) {
						value -= threshold;
						value *= scalar;
						value += threshold;
					}
					else if (value < -threshold){
						value += threshold;
						value *= scalar;
						value -= threshold;
					}
				}
				else if constexpr (std::is_unsigned_v<typename Dtype::Type>) {
					if (value > threshold) { //unsigned value
						value -= threshold;
						value *= scalar;
						value += threshold;
					}
				}
				else {
					static_assert(struct_assert<Asserts::AssertFalseWith<Dtype>>, "expected a signed or unsigned type");
				}
			}
		};
	}

}