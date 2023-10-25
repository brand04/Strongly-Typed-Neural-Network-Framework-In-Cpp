#pragma once
#include "../forward_declaration.h"
#include "partial_factor_shape.h"

namespace NN {
	namespace Shapes {
		namespace Operations {


			template<typename ShapeReversedT, typename PFShapeT>
			struct shapeToPartialFactorShape {};


			//Case 1: initial configuration (reversed Shape<...> and no partial factors
			template<unsigned int C0, unsigned int ...CS>
			struct shapeToPartialFactorShape<Shape<C0, CS...>, void> {
				using type = shapeToPartialFactorShape<Shape<CS...>, PartialFactorShape<C0>>::type;
			};

			//case 3: 1 parameter left - which is dropped (we aren't calculating volume, instead the volumes of every subShape (with the lowest order duplicated as above, to account for both the / and % operations needed in unflatten)
			template<unsigned int C0, unsigned int ...PS>
			struct shapeToPartialFactorShape < Shape<C0>, PartialFactorShape<PS...>> {
				using type = PartialFactorShape<PS...>;
			};


			//case 2: multiple parameters left, multiple parameters consumed
			template<unsigned int C0, unsigned int ...CS, unsigned int P0, unsigned int ...PS>
			struct shapeToPartialFactorShape < Shape<C0, CS...>, PartialFactorShape<P0, PS...>> {
				using type = shapeToPartialFactorShape < Shape<CS...>, PartialFactorShape<C0* P0, P0, PS...>>::type;
			};



			//case 5: Single length shape - no partial factors - use the single value
			template<unsigned int C0>
			struct shapeToPartialFactorShape<Shape<C0>, void> {
				using type = EmptyPartialFactorShape;
			};

		}
	}
}