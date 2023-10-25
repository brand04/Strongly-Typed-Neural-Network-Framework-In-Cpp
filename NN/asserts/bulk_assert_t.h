#pragma once
#include "./assert_t.h"

namespace NN {
	namespace Asserts {

		/// <summary>
		/// Runs an assertion on multiple template parameters
		/// </summary>
		/// <typeparam name = "Assertion"> An assertion taking a single type parameter</typeparam>
		/// <typeparam name="...TestTs">A pack of types to be used to instanciate the Assertion</typeparam>
		template<template<typename> typename Assertion, typename ... TestTs>
		struct BulkAssert : Assert<(Assertion<TestTs>::stassert() && ...)> {};
	}
}