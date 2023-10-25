#pragma once
#include "../../concepts/layer_concept.h"
namespace NN {
	namespace Sequences {

		//use a fold expression to get the last of a template pack
		template<Layers::Layer ... Layers>
		using getLast = typename decltype ((std::type_identity<Layers>{}, ...))::type;
	}
}