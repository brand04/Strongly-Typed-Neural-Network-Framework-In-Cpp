#pragma once
#include "device.h"
#include "gpu_base.h"

#include "../helpers/fixed_string.h"
#include "../helpers/compiler_operations/value_to_string_t.h"
#include "../helpers/string_collection.h"
namespace NN {
	namespace Devices {
		 
		template<const unsigned int deviceNumber> struct CUDA : GPUBase {
			static constexpr unsigned int deviceId = deviceNumber;
			static constexpr const FixedString string = StringCollection("CUDA Device ", Helpers::valueToString<deviceNumber>::string).fix();
		};

	}


}
