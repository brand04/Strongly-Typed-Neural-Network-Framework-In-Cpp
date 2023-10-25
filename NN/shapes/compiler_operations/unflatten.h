#pragma once
#include "../forward_declaration.h"
#include "partial_factor_shape.h"
namespace NN {
	namespace Shapes {
		namespace Operations {


			/*
			Unflatten operation - takes a shape, and flattened index, and returns the subShape with that unique id
			*/
			template<typename PartialFactorShapeT, const unsigned int, typename ShapeT>
			struct unflattener;

			//case 1: general case - multiple values left in partial factors
			template<const unsigned int P0, const unsigned int ...PS, const unsigned int ...SS, const unsigned int modulo>
			struct unflattener < PartialFactorShape<P0, PS...>, modulo, Shape<SS...>> {
				using type = unflattener < PartialFactorShape<PS...>, modulo% P0, Shape<SS..., modulo / P0>>::type;
			};

			//case 2: 1 value left in PartialFactors, so add the division and remainer to the shape
			template<const unsigned int P0, const unsigned int ...SS, const unsigned int modulo>
			struct unflattener<PartialFactorShape<P0>, modulo, Shape<SS...>> {
				using type = Shape<SS..., modulo / P0, modulo% P0>;
			};

			//case 3: initial case, construct the first shape
			template<const unsigned int P0, const unsigned int ...PS, const  unsigned int modulo>
			struct unflattener<PartialFactorShape<P0, PS...>, modulo, void> {
				using type = unflattener<PartialFactorShape<PS...>, modulo% P0, Shape<modulo / P0>>::type;
			};

			//special case, no partial factors
			template<const unsigned int modulo>
			struct unflattener<EmptyPartialFactorShape, modulo, void> {
				using type = Shape<modulo>;
			};
		}
	}
}