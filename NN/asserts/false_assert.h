#pragma once
#include "./assert_t.h"
#include <type_traits>

namespace NN {
	namespace Asserts {

		/// <summary>
		/// Always fails a static_assert but only at instantiation-time
		/// </summary>
		/// <typeparam name="T">The dependent type - possibly displayed in compiler error output</typeparam>
		template<typename T>
		struct AssertFalseWith : Assert<std::is_same_v<T, std::type_identity<T>>> {
			static_assert(std::is_same_v<T,std::type_identity<T>>, "asserted to be False");
		};

	}
}