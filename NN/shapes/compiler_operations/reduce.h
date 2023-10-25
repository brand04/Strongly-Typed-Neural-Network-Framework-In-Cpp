#pragma once
#include "../forward_declaration.h"
#include "../empty_shape.h"


namespace NN {
	namespace Shapes {
		namespace Operations {

			/*
			Reduces the shape by removing the highest dimension	(reducing dimension by 1)
			*/

			template<typename ShapeT>
			struct reducer;

			template<const unsigned int C0, const unsigned int ...CS>
			struct reducer<Shape<C0, CS ... >> {
				using type = Shape<CS...>;
			};

			//SPECIAL CASE: Single length shape
			template<const unsigned int C0>
			struct reducer<Shape<C0>> {
				using type = EmptyShape;
			};

		}
	}
}