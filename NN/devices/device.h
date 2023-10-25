#pragma once
#include <concepts>
#include "../helpers/fixed_string.h"
namespace NN {
	namespace Devices {

		struct DeviceBase {};

		template<typename D>
		concept Device = std::is_base_of_v<DeviceBase, D>;/*&& requires {
			D::string;
			D::string::size;
		} && std::same_as<FixedString<D::string::size>, decltype(D::string)>; */
	}
}