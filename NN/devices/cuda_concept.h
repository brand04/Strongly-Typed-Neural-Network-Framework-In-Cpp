#pragma once
#include "cuda_base.h"
#include <concepts>
namespace NN {
	namespace Devices{
		
		template<typename T>
		concept CUDADevice = requires {
			T::deviceId;
			std::is_base_of_v<CUDABase, T>;
		};

	}
}