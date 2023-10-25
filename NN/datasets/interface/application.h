#pragma once
#include "../../shapes/shape_concept.h"
#include "../../references/includes.h"
#include "../../devices/cpu.h"
namespace NN {
	namespace Datasets {
		namespace Interfaces {
			//Dataset for training
			template<typename T, Shapes::IsShape InputShape>
			class Application {
			private:
				virtual void getNext(unsigned long runId, T* input) = 0;

			public:
				virtual void get(unsigned long runId, References::MutableShapedDeviceReference<T, InputShape, Devices::CPU> input) {
					getNext(runId, input);
				}
			};
		}
	}
}