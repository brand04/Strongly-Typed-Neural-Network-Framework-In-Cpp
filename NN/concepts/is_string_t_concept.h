#pragma once
#include <concepts>
#include "../helpers/string_t.h"
#include "../helpers/string_literal.h"
namespace NN {
	namespace Concepts {

		//concept of a StringT type
		template<typename String>
		concept IsStringT = requires {
			String::length;
			std::same_as<const char(&)[String::length], decltype(String::value)>;
			std::same_as<StringLiteral<String::length>, decltype(String::string)>;

		};
	}
}