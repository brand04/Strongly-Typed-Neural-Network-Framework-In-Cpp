#pragma once
#include "../string_t_forward_declaration.h"
#include "../../shapes/shape_concept.h"
namespace NN {
	namespace Helpers {
		namespace CompilerOperations {
			/// <summary>
			/// Convert an unsigned value to a StringT
			/// </summary>
			/// <typeparam name="T">The StringT being generated</typeparam>
			/// <typeparam name="N">The value</typeparam>
			template<size_t N, typename T>
			struct valueToStringT;

			template<size_t N>
			struct valueToStringT<N, void> {
				using type = valueToStringT < N / 10, StringT<("0123456789"[N % 10]) >> ::type;
			};
			//recursive case
			template<unsigned int N, const char ... chars>
			struct valueToStringT<N, StringT<chars...>> final {
				using type = valueToStringT < N / 10, StringT<("0123456789"[N % 10]), chars...>>::type;

			};

			//base case
			template<const char ... chars>
			struct valueToStringT<0, StringT<chars...>> final {
				using type = StringT<chars...>;
			};

		
		}

		template<size_t N>
		using valueToString = CompilerOperations::valueToStringT<N, void>::type;
	}
}