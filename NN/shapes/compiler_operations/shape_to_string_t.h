#pragma once
#include "../../helpers/compiler_operations/value_to_string_t.h"
#include "../../helpers/string_t.h"

#include "../forward_declaration.h"
namespace NN {
	namespace Shapes {
		namespace Operations {


			template<typename ShapeT>
			struct shapeToStringT;

			//recursive case
			template<unsigned int component0, unsigned int ...components>
			struct shapeToStringT<Shape<component0, components...>> {
				using type = Helpers::CompilerOperations::valueToStringT<component0, void>::type::appendChar<'x'>::append<
					shapeToStringT<Shape<components...>>::type
				>;
				
			};

			//base case
			template<unsigned int component0>
			struct shapeToStringT<Shapes::Shape<component0>> {
				using type = Helpers::CompilerOperations::valueToStringT<component0, void>::type;

			};
		}
	}
}
