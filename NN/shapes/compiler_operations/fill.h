#pragma once
#include "../forward_declaration.h"

namespace NN {
	namespace Shapes {
		namespace Operations {


			/*
			Fill a shape with a constant value
			*/
			template<typename ShapeT, const unsigned int constant, const unsigned int dimension>
			struct filler;



			//Case 1: final case
			template<const unsigned int constant, const unsigned int ...Components>
			struct filler<Shape<Components...>, constant, 0> {
				using type = Shape<Components...>;
			};

			//case 2: general case
			template<const unsigned int constant, const unsigned int dimension, const unsigned int ...Components>
			struct filler<Shape<Components...>, constant, dimension> {
				using type = filler<Shape<constant, Components...>, constant, dimension - 1>::type;
			};

			//case 3: inititial case
			template<const unsigned int constant, const unsigned int dimension>
			struct filler<void, constant, dimension> {
				using type = filler<Shape<constant>, constant, dimension - 1>::type;
			};
		}
	}

}