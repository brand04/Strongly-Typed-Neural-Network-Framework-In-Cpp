#pragma once
#include "./assert_t.h"
#include "./layer_asserts.h"
#include <concepts>
#include <type_traits>
namespace NN {
	namespace Asserts {
		/// <summary>
		/// Runs a static assertion, when applied to struct_assert, that asserts that ActivationFunction has functions ::apply and ::derivative that both return and take a value of 'Type'
		/// </summary>
		/// <typeparam name="ActivationFunction">The Activation Function</typeparam>
		/// <typeparam name="Type">The Type to ensure the Activation Function works with</typeparam>
		template< typename ActivationFunction, typename Type>
		struct AssertExistsActivationFunctionFor : Assert <std::is_same_v<decltype(ActivationFunction::apply(std::declval<Type>())), Type>&& std::is_same_v<decltype(ActivationFunction::derivative(std::declval<Type>())), Type>> {
			static_assert(std::is_same_v<decltype(ActivationFunction::apply(std::declval<Type>())),Type> && std::is_same_v<decltype(ActivationFunction::derivative(std::declval<Type>())), Type> , "No Activation Function exists for the type specified - must be an type with static functions ::apply and ::derivative that are both function from Type -> Type");
		};

		/// <summary>
		/// Rnus a static assertion, when applied to struct_assert, that asserts that ActivationFunction has functions ::apply and ::derivative that both return and take a value of the underlying type of Dtype
		/// </summary>
		/// <typeparam name="Dtype">The wrapper around the compatible type to assert for</typeparam>
		/// <typeparam name="ActivationFunction">The Activation Function</typeparam>
		template<typename ActivationFunction, typename Dtype>
		struct AssertActivationFunction : Assert < AssertDtype<Dtype>::stassert() & AssertExistsActivationFunctionFor<ActivationFunction,typename Dtype::Type>::stassert()> {};

	
	}
}