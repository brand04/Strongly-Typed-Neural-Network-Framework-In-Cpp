#pragma once
#include "cpu.h"
#include <concepts>
namespace NN {
	namespace Devices {
		template<typename D>
		concept CPUDevice =	std::same_as<D, CPU>;
		
	}
}