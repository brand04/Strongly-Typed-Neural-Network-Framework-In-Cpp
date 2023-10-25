#pragma once
#include <type_traits>

namespace NN {
	namespace Helpers {
		template<size_t n, typename dummy = std::make_index_sequence<n> >
		struct IndexPackImpl;

		template<size_t n, size_t ... dummies>
		struct IndexPackImpl<n, std::index_sequence<dummies...>> {
			//declare a function with void* n times, and then the type to match to
			template<typename T>
			static T matchFunc(static_cast<void*>(dummies)..., std::type_identity<T>*);
		};

		template<size_t n, typename ...  pack>
		using getNthFrom = decltype(IndexPackImpl<n>::matchFunc(static_cast<std::type_identity<pack>*>(0)...));

	}
}