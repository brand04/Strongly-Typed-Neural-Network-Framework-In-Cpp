#pragma once
#include "device.h"
#include "../helpers/fixed_string.h"
namespace NN {
	namespace Devices {
		struct CPU : DeviceBase {
			static constexpr const FixedString string = FixedString("CPU");
		};
	}
}