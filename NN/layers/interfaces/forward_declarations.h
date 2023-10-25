#pragma once
#include "../../dtypes/concept.h"
#include "../../shapes/shape_concept.h"
#include "../../devices/device.h"
namespace NN {
	namespace Layers {
		namespace Interfaces {
			template<typename U, typename InputShape, typename InputDevice, typename V, typename OutputShape, typename OutputDevice, const unsigned int threads>
			class ILayer;
		}
	}
}