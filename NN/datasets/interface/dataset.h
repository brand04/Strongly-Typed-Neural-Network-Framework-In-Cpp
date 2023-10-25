#pragma once
#include "../../shapes/shape_concept.h"
#include "../../references/includes.h"
#include "../../devices/cpu.h"
namespace NN {
	namespace Datasets {
		namespace Interfaces {
			//Dataset for training
			template<typename U, typename V, Shapes::IsShape InputShape, Shapes::IsShape OutputShape>
			class Dataset {
			private:
				virtual void getTest(unsigned long runId, U* input, V* output) = 0;
				virtual void getTrainingSample(unsigned long runId, U* input, V* output) = 0;

			public:
				virtual void train(unsigned long runId, References::MutableShapedDeviceReference<U, InputShape, Devices::CPU> input, References::MutableShapedDeviceReference<V, OutputShape, Devices::CPU> output) { //virtual to allow potential usage of device
					//load data into memory for network - implicit cast to pointers
					getTrainingSample(runId, input, output);
				}

				virtual void test(unsigned long runId, References::MutableShapedDeviceReference<U, InputShape, Devices::CPU> input, References::MutableShapedDeviceReference<V, OutputShape, Devices::CPU> output) {
					getTest(runId, input, output);
				}
			};
		}
	}
}