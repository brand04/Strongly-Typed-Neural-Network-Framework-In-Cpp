#pragma once
#include "../string_t_forward_declaration.h"

namespace NN {
	namespace Helpers {
		namespace CompilerOperations {
			
			/// <summary>
			/// Appends a StringT to a StringT (concatenation)
			/// </summary>
			/// <typeparam name="LeftT"></typeparam>
			/// <typeparam name="RightT"></typeparam>
			template<typename LeftT, typename RightT>
			struct Appender;

			template<char ... lefts, char ... rights>
			struct Appender<StringT<lefts...>, StringT<rights...>> {
				using type = StringT<lefts..., rights...>;
			};
		}
	}
}