#pragma once
#include "./assert_t.h"
#include "../dtypes/concept.h"
#include "../functions/map_functions/mappable.h"
#include "./type_asserts.h"
#include <type_traits>
#include <stdlib.h>
namespace NN {
	namespace Asserts {
		/// <summary>
		/// Asserts that the Function has a ::apply function taking the underlying type of the datatype
		/// </summary>
		/// <typeparam name="Function">Function type to test</typeparam>
		/// <typeparam name="Dtype">The Dtype to test whether it is applicable to Function</typeparam>
		template<typename Function, typename Dtype>
		struct AssertMappableOn : Assert<struct_assert<AssertDtype<Dtype>>&& MapFunctions::Mappable<Function, Dtype>> {
			template<typename T>
			using ref = T&;
			static_assert(struct_assert<AssertDtype<Dtype>>&& MapFunctions::Mappable<Function, Dtype>, "Expected a function that can receive values of type Dtype");
		
		};
	}

}
