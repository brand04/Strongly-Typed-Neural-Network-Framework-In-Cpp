#pragma once
#include "../forward_declaration.h"

namespace NN {
	namespace Shapes {
		namespace Operations {


			/*
			Struct to subtract two shapes component-wise, producing a Shape Output
			*/
			template<typename ShapeL, typename ShapeR, typename ShapeO, bool orZero = false>
			struct subtracter;

			//case 1: general case - multiple params left in shapes
			template<const unsigned int L0, const unsigned int ...LS, const unsigned int R0, const unsigned int ...RS, const unsigned int ...OS, bool orZero>
			struct subtracter<Shape<L0, LS...>, Shape<R0, RS...>, Shape<OS...>, orZero> {
				static_assert(L0 >= R0 || orZero, "Attempting to subtract, but would cause negative axis");
				using type = subtracter<Shape<LS...>, Shape<RS...>, Shape<OS..., (L0 >= R0 ? L0 - R0 : 0)>, orZero>::type;
			};

			//case 2: initial case for multiple - length shapes
			template<const unsigned int L0, const unsigned int ...LS, const unsigned int R0, const unsigned int ...RS, bool orZero>
			struct subtracter<Shape<L0, LS...>, Shape<R0, RS...>, void, orZero> {
				static_assert(L0 >= R0 || orZero, "Attempting to subtract, but would cause negative axis");
				using type = subtracter<Shape<LS...>, Shape<RS...>, Shape< (L0 >= R0 ? L0 - R0 : 0)>, orZero>::type;
			};

			//case 3: final case for multiple-length shapes
			template<const unsigned int L0, const unsigned int R0, const unsigned int ...OS, bool orZero>
			struct subtracter<Shape<L0>, Shape<R0>, Shape<OS...>, orZero> {
				static_assert(L0 >= R0 || orZero, "Attempting to subtract, but would cause negative axis");
				using type = Shape<OS..., (L0 >= R0 ? L0 - R0 : 0)>;
			};

			//case 4: only case for single length shapes
			template<const unsigned int L0, const unsigned int R0, bool orZero>
			struct subtracter<Shape<L0>, Shape<R0>, void, orZero> {
				static_assert(L0 >= R0 || orZero, "Attempting to subtract, but would cause negative axis");
				using type = Shape<(L0 >= R0 ? L0 - R0 : 0)>;
			};
		}
	}
}