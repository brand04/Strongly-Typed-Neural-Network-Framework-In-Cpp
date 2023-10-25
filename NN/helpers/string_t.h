#pragma once
#include "./string_t_forward_declaration.h"
#include "./compiler_operations/append.h"
#include "./compiler_operations/value_to_string_t.h"
#include "./compiler_operations/static_literal_to_string_t.h"
#include "./string_literal.h"
namespace NN {

	/// <summary>
	/// Converts an unsigned value to a StringT
	/// </summary>
	/// <typeparam name="n">The value to convert</typeparam>
	template<size_t n>
	using toStringT = Helpers::CompilerOperations::valueToStringT<n, void>::type;

	/*
	A type-level string type
	The string is represented by a template parameter pack of chars
	used to convert types such as Shape<...> to a string
	*/
	template<const char ... chars>
	struct StringT {
		static constexpr unsigned int length = sizeof...(chars);
		static constexpr const char (value)[sizeof...(chars)+1] = {chars ..., '\0'};
		static constexpr const StringLiteral<sizeof...(chars) + 1> string = StringLiteral(value);
		
		//append one string T to another
		template<typename Other>
		using append = Helpers::CompilerOperations::Appender<StringT<chars...>, Other>::type;

		template<const char C>
		using appendChar = StringT<chars..., C>;

		
		//another alias for converting from a number to a StringT
		template<size_t n>
		using appendNumber = append<typename toStringT<n>>;
	};

	/// <summary>
	/// Converts a static constexpr char[]  to a type-level string represented by StringT<...>
	/// </summary>
	/// <typeparam name="StaticLiteral">The static constexpr char[]</typeparam>
	template<auto& StaticLiteral>
	using asStringT = typename Helpers::CompilerOperations::fromLiteralImpl<sizeof(StaticLiteral)/sizeof(char), 0, StaticLiteral>::type; //use sizeof to deduce the size of the char array - messy but functional - divide by sizeof(char) incase sizeof(char)!=1

	

	
}