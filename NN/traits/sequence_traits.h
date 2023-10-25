#pragma once
#include "../sequences/forward_declaration.h"
#include "../sequences/compiler_operations/duplicate.h"
#include "../concepts/layer_concept.h"
//provides sequencing operations

namespace NN {
	namespace Traits {

		/// <summary>
		/// Defines primitive operations for sequences such as append, prepend and duplicate
		/// </summary>
		/// <typeparam name="SequenceT">The type of the sequence to generate</typeparam>
		/// <typeparam name="...Layers">A paramater pack of layers</typeparam>
		template<template<typename...> typename SequenceT, Layers::Layer ... Layers>
		struct PrimitiveSequenceOperations;

		//single-layer case
		template<template<typename...> typename SequenceT, Layers::Layer Layer>
		struct PrimitiveSequenceOperations<SequenceT,Layer> {

			template< typename ... OtherLayers>
			using append = SequenceT<Layer, OtherLayers...>;

			template<typename ... OtherLayers>
			using prepend = SequenceT<OtherLayers..., Layer>;

			template< const unsigned int n>
			using duplicate = Sequences::CompilerOperations::duplicate<SequenceT, n, Layer>::type;


		};

		//multi-layer case
		template<template<typename...> typename SequenceT, Layers::Layer ... Layers>
		struct PrimitiveSequenceOperations {

			template<typename ... OtherLayers>
			using append = SequenceT<Layers..., OtherLayers...>;

			template<typename ... OtherLayers>
			using prepend = SequenceT<OtherLayers..., Layers...>;

			template<const unsigned int n>
			using duplicate = Sequences::CompilerOperations::duplicate<SequenceT, n, SequenceT<Layers...>>::type;


		};



		//Provides Sequencing operations to type based on its contained layers

		/// <summary>
		/// Provides common operations for any varients of a sequential sequence
		/// </summary>
		/// <typeparam name="SequenceT"> The type of the sequence to generate</typeparam>
		/// <typeparam name="...Layers">The layers contained within the current sequence</typeparam>
		template<template<typename...> typename SequenceT, Layers::Layer ... Layers>
		struct SequenceTraits : PrimitiveSequenceOperations<SequenceT, Layers...> {
			//can add any more complicated operations here
			
		};
		

	}
}
