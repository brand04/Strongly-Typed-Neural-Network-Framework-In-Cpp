#pragma once
#include "../network/launch_parameters/includes.h"
#include "../traits/trait_get.h"
namespace NN {
	namespace Traits {

		/// <summary>
		/// Defines the correct launch parameter types for the netwwork
		/// </summary>
		/// <typeparam name="LayerT">The Wrapped Layer Type</typeparam>
		template<typename LayerT>
		struct NetworkTraits {

			using TrainingParameters = NN::LaunchParameters::TrainingLaunchParameters<typename Traits::getLayerTraits<LayerT>::OutputType>;
			using TestParameters = NN::LaunchParameters::TestLaunchParameters<typename Traits::getLayerTraits<LayerT>::InputType,
				typename Traits::getLayerTraits<LayerT>::OutputType,
				typename Traits::getLayerTraits<LayerT>::InputShape,
				typename Traits::getLayerTraits<LayerT>::OutputShape
			>;

			using LaunchParameters = NN::LaunchParameters::LaunchParameters<typename Traits::getLayerTraits<LayerT>::OutputType, typename Traits::getLayerTraits<LayerT>::OutputShape>;
		};
	}
}