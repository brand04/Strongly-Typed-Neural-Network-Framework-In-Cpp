#pragma once
#include "../traits/trait_get.h"
#include "../traits/layer_traits.h"
#include "../concepts/layer_trait_concept.h"

namespace NN {
	namespace Layers {

		//A concept defining the various types and constants required for an object to be considered a layer
		//this allows a more concise wrapping and manipulations of layers into sequences via automatic type deductions
		template<typename L>
		concept Layer = LayerTrait<Traits::getLayerTraits<L>>;
	}
}