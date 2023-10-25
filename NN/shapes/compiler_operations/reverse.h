#pragma once
#include "../forward_declaration.h"

namespace NN {
	namespace Shapes {
		namespace Operations {


			//forward declaration of reversing
			template<typename LShape, typename RSHape>
			struct reverser;

			//intitialize a left shape to accumulate the reversed values
			template<typename ShapeT>
			using reverse = reverser<void, ShapeT>::type;



			/**
			Reverse the order of shape axis (used so that the shape can be written in an array-like way rather than in reverse order (ie highest order dimension on the right, not the left)

			*/


			//Case 1: void, 1 axis
			template<unsigned int R>
			struct reverser<void, Shape<R>> {
				using type = Shape<R>;
			};

			//case 2: void, n axis
			template<unsigned int R, unsigned int ...RS>
			struct reverser<void, Shape<R, RS...>> {
				using type = reverser<Shape<R>, Shape<RS...>>::type;
			};

			//case 3: m axis, 1 axis
			template <unsigned int ...LS, unsigned int R>
			struct reverser<Shape<LS...>, Shape<R>> {
				using type = Shape<R, LS...>;
			};

			//case 4, m axis, n axis
			template< unsigned int ... LS, unsigned int R, unsigned int ... RS>
			struct reverser<Shape<LS...>, Shape<R, RS...>> {
				using type = reverser<Shape<R, LS...>, Shape<RS...>>::type;
			};

		}
	}
}