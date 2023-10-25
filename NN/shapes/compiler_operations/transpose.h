#pragma once
#include "../forward_declaration.h"
#include "../../asserts/assert_t.h"
#include "../../asserts/false_assert.h"
namespace NN {
	namespace Shapes {
		namespace Operations {


			/*
			* Allow transposition of a 2 dimensional shape if one of the dimensions is unit sized
			*
			* This is a safe operation as the memory locations are just a single line of memory, which can be read either as a column or a row
			*
			*/
			template<typename ShapeT>
			struct transposeShape;

			//if first dimension non-unit
			template<const unsigned int C0>
			struct transposeShape<Shape<C0, 1>> {
				using type = Shape<1, C0>;
			};

			//if second dimension non-unit
			template<const unsigned int C1>
			struct transposeShape<Shape<1, C1>> {
				using type = Shape<C1, 1>;
			};

			//special case: if both are 1 - causes a failure to partially order the two above specializations
			template<>
			struct transposeShape<Shape<1, 1>> {
				using type = Shape<1, 1>;
			};

			//Error path - does not match any of the required paths
			template<const unsigned int ...CS>
			struct transposeShape<Shape<CS...>> {
				static_assert(struct_assert<Asserts::AssertFalseWith<Shape<CS...>>>, "Invalid Shape to transpose");
			};

		}
	}
}