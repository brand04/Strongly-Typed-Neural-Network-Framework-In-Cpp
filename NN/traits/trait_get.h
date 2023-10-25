#pragma once

#include "./layer_traits.h"

namespace NN {
	namespace Traits {


		//Get the traits for a layer - [Input/Output][Shape/Device/Dtype/Type]
		template<typename LayerT>
		using getLayerTraits = typename LayerT::LayerTraits; //use the publically exposed type alias LayerT::LayerTraits - slightly breaks encapsulation but no better solution if a child type needs to be able to redeclare its LayerTraits (Such as with specialized versions of layers like Linear if it uses Overlay)
	}

}