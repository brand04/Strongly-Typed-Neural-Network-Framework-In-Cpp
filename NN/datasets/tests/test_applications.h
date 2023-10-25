#pragma once
#include "../interface/application.h"
#include "../../shapes/includes.h"
namespace NN {
	namespace Datasets {
		namespace Tests {
			template<typename T, Shapes::IsShape Shape = Shapes::unit<1>>
			class RandomTest : public Interfaces::Application<T,Shape> {
			public:
				virtual void getNext(unsigned long runId, T* input) override {
					for (int i = 0; i < Shape::volume; i++) {
						T val = (T)(rand() % 100) / 10;
						input[i] = val;
					}
				}
			};
		}
	}
}