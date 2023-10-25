#pragma once
#include <concepts>
namespace NN {
	namespace Layers {
		template<typename TraitT>
		concept LayerTrait = requires {

			Shapes::IsShape<typename TraitT::InputShape>;
			Shapes::IsShape<typename TraitT::OutputShape>;
			Devices::Device<typename TraitT::InputDevice>;
			Devices::Device<typename TraitT::OutputDevice>;

			Dtypes::Dtype<typename TraitT::InputDtype>;
			Dtypes::Dtype<typename TraitT::OutputDtype>;
			std::same_as<typename TraitT::InputDtype::Type, typename TraitT::InputType>;
			std::same_as<typename TraitT::OutputDtype::Type, typename TraitT::OutputType>;

			std::same_as<const unsigned int, decltype(TraitT::nThreads)>;
		};
	}
}
