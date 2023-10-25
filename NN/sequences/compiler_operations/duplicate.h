#pragma once
#include "../forward_declaration.h"

namespace NN {
	namespace Sequences {
		namespace CompilerOperations {

			//duplicate a layer n times, resulting in a Sequence of length n+1
			template<template<typename...> typename SequenceT,unsigned int n, Layers::Layer ... Layers>
			struct duplicate;
			//base case
			template<template<typename...> typename SequenceT, Layers::Layer... Layers>
			struct duplicate<SequenceT,0, Layers...> {
				using type = SequenceT<Layers...>;
			};

			//recursive case
			template<template<typename...> typename SequenceT, unsigned int n, Layers::Layer Layer0, Layers::Layer ... Layers> requires (n!=0)
			struct duplicate<SequenceT,n, Layer0, Layers...> {
				using type = duplicate<SequenceT,n - 1, Layer0, Layer0, Layers...>::type;
			};

			

		}
	}
}