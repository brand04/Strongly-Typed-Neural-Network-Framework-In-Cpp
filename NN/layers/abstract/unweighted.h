#pragma once
#include "./layer_box.h"
namespace NN {
	namespace Layers {
		namespace Abstract {

			template<typename InputDtype, typename InputShape, typename InputDevice, typename OutputDtype, typename OutputShape, typename OutputDevice, unsigned int threads>
			class UnweightedLayer : public LayerBox<InputDtype, InputShape, InputDevice, OutputDtype, OutputShape, OutputDevice, threads> {
				using InputType = typename InputDtype::Type;
				using OutputType = typename OutputDtype::Type;
			public:
				static constexpr StringLiteral name = StringLiteral("Unweighted Layer");

				void threadModifyWeights(
					References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
					References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
					Threads::ThreadContext& threadContext
				){} //no weights to modify

				template<typename LayerT>
				static void saveAsBytes(LayerT& layer, std::ofstream& file, Threads::ThreadContext& ctx) {
					//no weights - no action required
				}

				template<typename LayerT>
				static void readFromBytes(LayerT& layer, std::ifstream& file, Threads::ThreadContext& ctx) {
					//no weights - no action required
				}
				using Abstract::LayerBox<InputDtype, InputShape, InputDevice, OutputDtype, OutputShape, OutputDevice, threads>::LayerBox;


				void setWeights(uint8_t*& handle, const Threads::ThreadContext& threadContext) {} //no action required
				void initializeWeights(const Threads::ThreadContext& threadContext) {} //no action required
				template<typename LayerT>
				void displayWeights(std::stringstream& s, std::string prepend, const Threads::ThreadContext& threadContext) const {} //no action required
			};
		}
	}
}