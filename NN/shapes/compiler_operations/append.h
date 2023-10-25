#pragma once
#include "../forward_declaration.h"

namespace NN {
	namespace Shapes {
		namespace Operations {

		/*
		* Append to a  Shape by adding an extra axis of a specified size (increasing the dimension by 1) - differs from expand as it can cause problems with memory since the axis is added at the lowest, not highest, dimension
		*/
			template<typename ShapeT, const unsigned int C>
			struct appendTo;

			template<const unsigned int ...CS, const unsigned int C>
			struct appendTo<Shape<CS...>, C> {
				using type = Shape<C, CS...>;
			};


		}
	}

}