#pragma once
#include "./layer_asserts.h"

namespace NN {
	namespace Asserts {

		/// <summary>
		/// Asserts that the sequenced layers can each transition to the next layer
		/// </summary>
		/// <typeparam name="...Ls">The Sequences of layers</typeparam>
		template<typename ... Ls>
		struct AssertSequence;

		template<typename L0, typename L1, typename ... Ls>
		struct AssertSequence<L0, L1, Ls...> : Assert<> {
			static consteval bool stassert() {
				return struct_assert<AssertLegalTransition<L0, L1>> && struct_assert<AssertSequence<L1, Ls...>>; //performs static_asserts with more helpful compiler messages
			}
		};

		template<typename Ln>
		struct AssertSequence<Ln> : Assert<> {}; //true by default
	}
}