#pragma once
#include <type_traits>
#include <concepts>
namespace NN {
	namespace Functions {
		namespace InvertibleLinearTransformations {

			/// <summary>
			/// Concept defining Function as an InvertibleLinearTransformation on T
			/// Note that this does not check that the laws for the LinearTransformation apply - namely
			/// Function<T>::apply(x) + Function<T>::apply(y) = Function<T>::apply(x+y) and Function<T>::apply(kx) = kFunction<T>::apply(x), and the same for the inverse which will always be true if it is an invertible linear transformation
			/// These restrictions allow the function to be used during the forward pass and then inversed during the backward pass
			/// </summary>
			template<template<typename> typename Function, typename T>
			concept InvertibleLinearTransformation = requires {
				typename Function<T>;
				std::is_void_v<decltype(Function<T>::apply(std::declval<T>()))>;
				std::is_void_v<decltype(Function<T>::inverse(std::declval<T>()))>;
			};
		}
	}
}