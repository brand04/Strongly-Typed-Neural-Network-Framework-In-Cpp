#pragma once
#include "../forward_declaration.h"

namespace NN {
	namespace Shapes {
		namespace Operations {

			//add to the head of the shape with a new higher-order axis

			template<typename ShapeT, const unsigned int expansion>
			struct expander;

			template<const unsigned int C0, const unsigned int ...CS, const unsigned int expansion>
			struct expander<Shape<C0, CS...>, expansion> {
				using type = Shape<expansion, C0, CS...>;
			};

		}
	}
}