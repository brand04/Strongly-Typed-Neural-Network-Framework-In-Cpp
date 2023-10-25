#pragma once
#include "./assert_t.h"
#include "./type_asserts.h"
namespace NN {
	namespace Asserts {

		/// <summary>
		/// Runs a static assertion, when passed to struct_assert<...>, that asserts the shape has a dimension of 1
		/// </summary>
		/// <typeparam name="Shape">A 1 dimensional shape</typeparam>
		template<typename Shape>
		struct AssertIsOneDimensional : Assert<AssertShape<Shape> && Shape::dimension == 1> {
			static_assert(Shape::dimension == 1, "Expected a 1 Dimensional Shape");
		};
	}
}