#pragma once
#include "../helpers/string_collection.h"
namespace NN {
	namespace Concepts {
		namespace Tests {
			template<typename T>
			struct IsStringCollectionTest {
				static const constexpr bool value = false;
			};

			template<typename ... Ts>
			struct IsStringCollectionTest<StringCollection<Ts...>> {
				static const constexpr bool value = true;
			};

			template<typename ... Ts>
			struct IsStringCollectionTest<const StringCollection<Ts...>> {
				static const constexpr bool value = true;
			};
		}

		template<typename T>
		static constexpr bool IsStringCollectionV = Tests::IsStringCollectionTest<T>::value;

		template<typename T>
		concept IsStringCollection = requires {
			IsStringCollectionV<T>;
		};
	}
}