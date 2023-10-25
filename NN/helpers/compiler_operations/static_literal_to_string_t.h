#pragma once
#include "../string_t_forward_declaration.h"
namespace NN {
	namespace Helpers {
		namespace CompilerOperations {

			/// <summary>
			/// Converts a static constexpr const char&[] to a StringT
			/// </summary>
			/// <typeparam name="String">The StringT being generated</typeparam>
			/// <typeparam name="n">The size of the const char&[]</typeparam>
			/// <typeparam name="index">The current index within the const char&[]</typeparam>
			/// <typeparam name="str">The const char&[] to copy into the StringT</typeparam>
			template<size_t n, size_t index, const char(&str)[n], typename String = StringT<>>
			struct fromLiteralImpl;

			template<size_t n, size_t index, const char(&str)[n], typename String> requires (index >= n - 1) //n-1 to not include null byte
				struct fromLiteralImpl<n, index, str, String> {
				using type = typename String::appendChar<str[index]>; //append the character
			};

			template<size_t n, size_t index, const char(&str)[n], typename String> requires (index < n - 1) //<n-1 to not include null byte
				struct fromLiteralImpl<n, index, str, String> {
				using type = typename fromLiteralImpl<n, index + 1, str, typename String::appendChar<str[index]>>::type; //append the char and move onto the next char
			};
		}
	}
}