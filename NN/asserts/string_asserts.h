#pragma once
#include "./assert_t.h"
#include "../helpers/fixed_string.h"
#include "../helpers/string_literal.h"
#include "../helpers/string_t.h"
#include <concepts>
namespace NN {
	namespace Asserts {

		//contains asserts for the different methods of string representations - StringT (a type-level char parameter pack), StringLiteral (a reference to a pre-existing sequence of chars - usually a string literal) and FixedString (an owned sequence of chars)

		
		
		/// <summary>
		/// Asserts that the template parameter supplied is of type FixedString<[some length]>, failing a static assert if not so and returning false when applied in struct_assert
		/// </summary>
		template<typename String>
		struct AssertIsFixedString : Assert < requires { String::size; std::same_as<decltype(String::size), size_t>; std::same_as<String, FixedString<String::size>> || std::same_as<String, const FixedString<String::size>>; } > {
			static_assert(requires { String::size; std::same_as<decltype(String::size), size_t>; std::same_as<String, FixedString<String::size>> || std::same_as<String, const FixedString<String::size>>; }, "Expected a FixedString");
		};

		///<summary>
		///Asserts that the template parameter is of type StringLiteral<[some length]>, failing a static assert if not so and returning false when applied in struct_assert
		///</summary>
		template<typename String>
		struct AssertIsStringLiteral : Assert < requires { String::size; std::same_as<decltype(String::size), size_t>; std::same_as<String, StringLiteral<String::size>> || std::same_as<String, const StringLiteral<String::size>>; } > {
			static_assert(requires { String::size; std::same_as<decltype(String::size), size_t>; std::same_as<String, StringLiteral<String::size>> || std::same_as<String, const StringLiteral<String::size>>; }, "Expected a StringLiteral - perhaps the StringLiteral(\"...\") was not included");
		};

		///<summary> 
		/// Asserts that the template parameter is of type StringLiteral<[some length]> or FixedString<[some length]>, failing a static assert if not so and returning false when applied in struct_assert
		/// </summary>
		template<typename String>
		struct AssertIsConstexprString : Assert < requires { typename AssertIsFixedString<String>; } || requires { typename AssertIsStringLiteral<String>; } > {
			static_assert(requires { typename AssertIsFixedString<String>; } || requires { typename AssertIsStringLiteral<String>; } , "Expected a StringLiteral or FixedString type");
		};

		/// <summary>
		/// Asserts tha the supplied template paramter is a StringCollection
		/// </summary>
		/// <typeparam name="StringCollectionT">The type to assert is a StringCollection</typeparam>
		template<typename StringCollectionT>
		struct AssertIsStringCollection : Assert<false> { //false path - primary specialization
			static_assert(struct_assert<Asserts::AssertFalseWith<StringCollectionT>>, "The supplied template argument is not a StringCollection");
		};

		template<typename ... StrTypes>
		struct AssertIsStringCollection<StringCollection<StrTypes...>> : Assert<true> {};

		template<typename ... StrTypes>
		struct AssertIsStringCollection<const StringCollection<StrTypes...>> : Assert<true> {};

		///<summary>
		///Asserts that the template parameter is of type StringT<char 1, char 2 , ... , char n-1, char n>, failing a static assert if not so and returning false when applied in struct_assert
		///</summary>
		template<typename StringT>
		struct AssertIsStringT : Assert<false>{ // false path - primary specialization
			static_assert(struct_assert<Asserts::AssertFalseWith<StringT>>, "The supplied template argument is not a StringT with a char paramater pack");
		};

		//partial specialization for the non-failing path
		template<char ... chars>
		struct AssertIsStringT<StringT<chars...>> : Assert<true>{};
	}
		
}