#pragma once
#include "store.h"
#include "training_store.h"
namespace NN {
	namespace Storage {

		//a struct containing three stores containing space to store output values, node deltas as well as expected values
		template<Dtypes::Dtype T, Shapes::IsShape Shape, const unsigned int threads, Devices::Device Device = Devices::CPU>
		struct TrainingOutputStore : public TrainingStore<T,Shape,threads,Device>{
			Store<T, Shape, threads, Device> expected;
		};

	}
}