#pragma once
#include <cuda_runtime.h>
#include <concepts>

namespace NN {
	namespace WeightModifiers {

		template<typename W, typename T>
		concept WeightModifier = requires {
			{ W::adjust(T(), T()) } -> ::std::same_as<T>;
		};
	}
}