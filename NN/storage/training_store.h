#pragma once
#include "store.h"
#include "../tensors/tensors.h"
namespace NN {
	namespace Storage {

		//a struct containing two stores containing space to store inter-layer values as well as node deltas for training
		template<Dtypes::Dtype T, Shapes::IsShape Shape, const unsigned int threads, Devices::Device Device = Devices::CPU>
		struct TrainingStore {
			Store<T, Shape, threads, Device> data;
			Store<T, Shape, threads, Device> deltas;

			Tensors::Tensor<T, Shape, threads, Device> makeTensor() {
				return Tensors::Tensor<T, Shape, threads, Device>(this);
			}

			TrainingStore() : data(), deltas() {}
		};



	}
}