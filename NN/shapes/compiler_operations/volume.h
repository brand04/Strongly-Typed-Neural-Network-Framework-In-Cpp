#pragma once
#include "../forward_declaration.h"

namespace NN {
	namespace Shapes {
		namespace Operations {

			template<typename ShapeT, const unsigned int V>
			struct getVolume {
				static const unsigned int value;
			};

			//partial specializations

			//recursive case
			template <const unsigned int C0, const unsigned int ... CS, const unsigned int V>
			struct getVolume<Shape<C0, CS...>, V> {
				static const unsigned int value = getVolume<Shape<CS...>, V* C0>::value;
			};

			//base case
			template <const unsigned int C0, const unsigned int V>
			struct getVolume<Shape<C0>, V> {
				static const unsigned int value = V * C0;
			};
		}
	}
}