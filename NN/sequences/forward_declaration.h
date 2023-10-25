#pragma once
#include "../concepts/layer_concept.h"
namespace NN {

	namespace Sequences {
		template<Layers::Layer ... Layers>
		class Subsequence;

		template<Layers::Layer ... Layers>
		class SequenceImpl;
	}

	template<Layers::Layer ... Layers>
	class Sequence;
}