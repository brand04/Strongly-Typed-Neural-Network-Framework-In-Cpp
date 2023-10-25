#pragma once
#include "./string_t.h"
#include "../asserts/string_asserts.h"
namespace NN {
	namespace Helpers {
		namespace CompilerOperations {

			template<size_t n, size_t index, const char(&str)[n], typename String = StringT<>>
			struct fromLiteralImpl;

			template<size_t n, size_t index, const char(&str)[n], typename String> requires (index >= n - 1) //n-1 to not include null byte
				struct fromLiteralImpl<n, index, str, String> {
				using type = typename String::appendChar<str[index]>;
			};

			template<size_t n, size_t index, const char(&str)[n], typename String> requires (index < n - 1) //<n-1 to not include null byte
				struct fromLiteralImpl<n, index, str, String> {
				using type = typename fromLiteralImpl<n, index + 1, str, typename String::appendChar<str[index]>>::type;
			};
		}
	}

	//from a static literal, construct a StringT (String at the type level)
	template<auto& StaticLiteral>
	using fromStaticLiteral = typename Helpers::CompilerOperations::fromLiteralImpl<sizeof(StaticLiteral), 0, StaticLiteral>::type; //use sizeof to deduce the size of the char array - messy but functional

}