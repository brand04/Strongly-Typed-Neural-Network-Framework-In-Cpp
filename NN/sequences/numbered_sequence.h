#pragma once
#include "../asserts/false_assert.h"
#include "../helpers/string_t.h"
#include "../helpers/string_literal.h"
#include "./sequence.h"
#include "../traits/sequence_traits.h"
#include "../layers/phantom_layers/name_prepender.h"
#include <tuple>
namespace NN {
	namespace Sequences {

		template<typename Numbers, typename...Layers>
		struct NumberedSequenceImpl {
			static_assert(struct_assert<Asserts::AssertFalseWith<Numbers>>, "Expected a std::index_sequence as the first template argument, and a sequence of layers as the proceeding ones");
		};

	}

	//in NN namespace
	template<typename ... LayerTs>
	using NumberedSequence = Sequences::NumberedSequenceImpl<std::make_index_sequence<sizeof...(LayerTs)>, LayerTs...>;

	namespace Sequences {
		/// <summary>
		/// A subsequence that incorporates each wrapped layer's index into it's name
		/// </summary>
		/// <typeparam name="...LayerTs">The sequence of Layers</typeparam>
		/// <typeparam name="...Numbers">An index sequence</typeparam>
		template<size_t ... Numbers, typename ... LayerTs>
		class NumberedSequenceImpl<std::index_sequence<Numbers...>, LayerTs...> : 
			public SequenceImpl<Layers::Phantom::NamePrepender<LayerTs, typename toStringT<Numbers>::append<StringT<'.', ' '>>>...>,

			public Traits::SequenceTraits<NumberedSequence, LayerTs...> {
			
		public:
			static constexpr StringLiteral name = StringLiteral("Numbered Sequence");
			using SequenceImpl<Layers::Phantom::NamePrepender<LayerTs, typename toStringT<Numbers>::append<StringT<'.', ' '>>>...>::SequenceImpl;


		};
	}

	


}