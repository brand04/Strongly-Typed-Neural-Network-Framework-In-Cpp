#pragma once
#include "../forward_declaration.h"
#include "partial_factor_shape.h"

/*
Flatten Operation (constexpr)
*/
namespace NN {
	namespace Shapes {
		namespace Operations {
			template<typename PartialFactorShapeT, typename SubShapeT, const unsigned int>
			struct flattener;

			//case 1: recursive case - at least one parameter on each
			template<const unsigned int P0, const unsigned int ...PS, const unsigned int C0, const unsigned int ...CS, const unsigned int flat>
			struct flattener < PartialFactorShape<P0, PS...>, Shape<C0, CS...>, flat> {
				static constexpr unsigned int value = flattener<PartialFactorShape<PS...>, Shape<CS...>, flat + (P0 * C0)>::value;
			};

			//case 2: base case - 2 values left in subshape, 1 value left in partialFactors
			template<const unsigned int P0, const unsigned int C0, const unsigned int C1, const unsigned int flat>
			struct flattener < PartialFactorShape<P0>, Shape<C1, C0>, flat> {
				static constexpr unsigned int value = flat + (C1 * P0) + C0;
			};

			//special case - no partial factors
			template<const unsigned int C0>
			struct flattener<EmptyPartialFactorShape, Shape<C0>, 0> {
				static constexpr unsigned int value = C0;
			};
		}
	}
}