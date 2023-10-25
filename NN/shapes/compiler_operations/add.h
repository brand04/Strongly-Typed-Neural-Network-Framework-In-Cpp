#pragma once
#include "../forward_declaration.h"


namespace NN {
	namespace Shapes {
		namespace Operations {


			/*
			Struct to add two shapes component-wise, producing a Shape Output using ::type
			*/
			template<typename ShapeL, typename ShapeR, typename ShapeO>
			struct adder;

			//case 1: general case - multiple params left in shapes
			template<const unsigned int L0, const unsigned int ...LS, const unsigned int R0, const unsigned int ...RS, const unsigned int ...OS >
			struct adder<Shape<L0, LS...>, Shape<R0, RS...>, Shape<OS...>> {
				using type = adder<Shape<LS...>, Shape<RS...>, Shape<OS..., L0 + R0>>::type;
			};

			//case 2: initial case for multiple-length shapes
			template<const unsigned int L0, const unsigned int ...LS, const unsigned int R0, const unsigned int ...RS>
			struct adder<Shape<L0, LS...>, Shape<R0, RS...>, void> {
				static_assert(sizeof...(LS) == sizeof...(RS), "Attempting to add together shapes of two different dimensions");
				using type = adder<Shape<LS...>, Shape<RS...>, Shape<L0 + R0>>::type;
			};

			//case 3: final case for multiple-length shapes
			template<const unsigned int L0, const unsigned int R0, const unsigned int ...OS>
			struct adder<Shape<L0>, Shape<R0>, Shape<OS...>> {
				using type = Shape<OS..., L0 + R0>;
			};

			//case 4: only case for single length shapes
			template<const unsigned int L0, const unsigned int R0>
			struct adder<Shape<L0>, Shape<R0>, void> {
				using type = Shape<L0 + R0>;
			};
		}
	}
}