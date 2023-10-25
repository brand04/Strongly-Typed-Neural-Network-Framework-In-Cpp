#pragma once

namespace NN {
	namespace Asserts {



		/*
		since concept compiler error messages ...aren't great, and we rely so heavily on compiler error messages since nearly all errors are compile-time,
		provide a set of structs that contain static_asserts with more helpful error messages - though still somewhat difficult to read it will at least highlight the troublesome types
		By splitting down the complex structures into simple 1 or 2 template parameter structs, the instantiations of the struct will contain precisely which arguments failed the assert and each instantiation above will provide more information about where specifically failed

		This is required due to the fact that static_asserts can only take a string literal so we cannot construct it from for example char (&)[] and we want the compiler messages

		In a nearby version this can likely be reduced to very few functions by making use of compile-time strings, once static_assert supports non-string-literal error messsages
		*/

		/*
		The base class of all custom asserts
		Provides a single static consteval bool stassert() that forces an evaluation of any static_asserts
		To return a custom value, the stassert function should be overriden, or the boolean template paramter should be filled with the assert
		*/
		template<bool ... Assertion>
		struct Assert;

		//take no assertion and return true regardless
		template<>
		struct Assert<> {
			static consteval bool stassert(void) {
				return true;
			}
		};


		//take the result of an assertion and return it from the assert function
		template<bool Assertion>
		struct Assert<Assertion> {
			static consteval bool stassert(void) {
				return Assertion;
			}
		};		
	}

	/// <summary>
	/// Given an Assertion type, instantiates it to evaluate any static assertions within and becomes the result of that static assertion (to inform the caller of the result)
	/// </summary>
	/// <typeparam name="Assertion">A type inheriting from Assert<bool> or a type that implements a static consteval bool Assertion::stassert() </typeparam>
	template<typename Assertion>
	static constexpr bool struct_assert = Assertion::stassert();

	
	
}