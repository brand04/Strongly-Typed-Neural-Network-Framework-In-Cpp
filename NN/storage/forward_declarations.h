#pragma once
#include "../shapes/shape_concept.h"
#include "../devices/includes.h"
#include "../dtypes/concept.h"
namespace NN {
	namespace Storage {

		template<Dtypes::Dtype T, Shapes::IsShape Shape, const unsigned int threads, Devices::Device Device>
		class Store;

		template<Dtypes::Dtype T, Shapes::IsShape Shape, const unsigned int threads, Devices::Device Device>
		struct TrainingStore;

		template<Dtypes::Dtype T, Shapes::IsShape Shape, const unsigned int threads, Devices::Device Device>
		struct TrainingOutputStore;
	}
}