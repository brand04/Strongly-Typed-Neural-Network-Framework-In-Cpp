#pragma once
#include "./i_layer.h"
namespace NN {
	namespace Layers {
		namespace Interfaces {

			/// <summary>
			/// Similar to ILayer, except has default implementations of functions related to weights such as saveAsBytes and displayWeights
			/// Used for layers without weights
			/// </summary>
			/// <typeparam name="InputDtype"></typeparam>
			/// <typeparam name="InputShape"></typeparam>
			/// <typeparam name="InputDevice"></typeparam>
			/// <typeparam name="OutputDtype"></typeparam>
			/// <typeparam name="OutputShape"></typeparam>
			/// <typeparam name="OutputDevice"></typeparam>
			/// <typeparam name="threads"></typeparam>
			template<typename InputDtype, typename InputShape, typename InputDevice, typename OutputDtype, typename OutputShape, typename OutputDevice, unsigned int threads>
			class ImmaterialLayer : public ILayer<InputDtype, InputShape, InputDevice, OutputDtype, OutputShape, OutputDevice, threads> {
				using Self = ImmaterialLayer<InputDtype, InputShape, InputDevice, OutputDtype, OutputShape, OutputDevice, threads>;
			public:

				template<typename LayerT>
				static void modifyWeights(LayerT& layer, Threads::ThreadContext& ctx) {}; //no weights

				template<typename LayerT>
				static void saveAsBytes(LayerT& layer, std::ofstream& file, Threads::ThreadContext& ctx) {}; 

				template<typename LayerT>
				static void readFromBytes(LayerT& layer, std::ifstream& file, Threads::ThreadContext& ctx) {}

				void displayWeights(std::stringstream& s, std::string prepend, Threads::ThreadContext& threadContext) {}

				void initializeWeights(Threads::ThreadContext& threadContext) {}


				void setWeights(uint8_t*& handle, Threads::ThreadContext& threadContext) {}
			};
		}
	}
}