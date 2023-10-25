#pragma once
#include <fstream>
#include "../layers/interfaces/i_layer.h"
#include "../storage/training_store.h"
#include "../concepts/layer_concept.h"
#include "../tensors/tensors.h"
#include "../references/includes.h"
#include "./compiler_operations/get_last.h"
#include "./compiler_operations/duplicate.h"
#include "./forward_declaration.h"
#include "../traits/trait_get.h"
#include "../traits/sequence_traits.h"
#include "./compiler_operations/assert_sequence.h"

namespace NN {
	namespace Sequences {


		//Provides a wrapper class around one or more layers that allows combining them together safely, whilst handling intermediatory storages
				//This class can be treated as its own standalone layer and as such can also be further combined into higher order sequences
				// 
				// makes use of the Layer concept, ensuring
				//types I_Dtype, I_Shape, I_Device, O_Dtype, O_Shape, O_Device and unsigned int NThreads exist and match the expected I/O types of the preceeding and succeeding layer

				//A sequence of layers that each pass its output to the next



				//recursive case
		template<Layers::Layer Layer0, Layers::Layer ... LayerTs>
		class Subsequence<Layer0, LayerTs...> : public Layers::Interfaces::ILayer<
			typename Traits::getLayerTraits<Layer0>::InputDtype,
			typename Traits::getLayerTraits<Layer0>::InputShape,
			typename Traits::getLayerTraits<Layer0>::InputDevice,
			typename Traits::getLayerTraits<
			getLast<Layer0, LayerTs...>
			>::OutputDtype,
			typename Traits::getLayerTraits<
			getLast<Layer0, LayerTs...>
			>::OutputShape,
			typename Traits::getLayerTraits<
			getLast<Layer0, LayerTs...>
			>::OutputDevice,
			Traits::getLayerTraits<Layer0>::nThreads
		>{
		private:
			//alias the traits
			using InputDtype = typename Traits::getLayerTraits<Layer0>::InputDtype;
			using InputShape = typename Traits::getLayerTraits<Layer0>::InputShape;
			using InputDevice = typename Traits::getLayerTraits<Layer0>::InputDevice;

			using InputType = typename Traits::getLayerTraits<Layer0>::InputType;

			using OutputDtype = typename Traits::getLayerTraits<getLast<Layer0, LayerTs...>>::OutputDtype;
			using OutputShape = typename Traits::getLayerTraits<getLast<Layer0, LayerTs...>>::OutputShape;
			using OutputDevice = typename Traits::getLayerTraits<getLast<Layer0, LayerTs...>>::OutputDevice;

			//local for this recursive depth in the struct
			using LocalOutputDevice = typename Traits::getLayerTraits<Layer0>::OutputDevice;
			using LocalOutputDtype = typename Traits::getLayerTraits<Layer0>::OutputDtype;
			using LocalOutputShape = typename Traits::getLayerTraits<Layer0>::OutputShape;

			using LocalOutputType = typename Traits::getLayerTraits<Layer0>::OutputType;

			static constexpr const unsigned int nThreads = Traits::getLayerTraits<Layer0>::nThreads;

			Storage::TrainingStore<LocalOutputDtype, LocalOutputShape, nThreads, LocalOutputDevice> sharedStore;
			Layer0 head;
			Subsequence<LayerTs...> tail; //recursive entry
		public:

			Subsequence(

				Tensors::Tensor<InputDtype, InputShape, nThreads, InputDevice> inputs,
				Tensors::Tensor<OutputDtype, OutputShape, nThreads, OutputDevice> outputs
			) :
				head(inputs, sharedStore.makeTensor()), //initialize the Layer for this recursive depth
				tail(sharedStore.makeTensor(), outputs) {
			} //recursively initialize subSequences, using the sharedStore as the next input

			Subsequence(Subsequence<Layer0, LayerTs...>&) = delete; //delete copy constructor
			Subsequence(Subsequence<Layer0, LayerTs...>&&) = delete; //keep move constructor



			void initializeWeights(Threads::ThreadContext& ctx) {
				head.initializeWeights(ctx);
				tail.initializeWeights(ctx);
			}

			void setWeights(uint8_t*& handle, Threads::ThreadContext& threadContext) {
				head.setWeights(handle, threadContext);
				tail.setWeights(handle, threadContext);
			}

			template<typename LayerT>
			void displayWeights(std::stringstream& s, std::string prepend, Threads::ThreadContext& threadContext) const {
				head.template displayWeights<Layer0>(s, prepend + "\t", threadContext);
				tail.template displayWeights<Subsequence<LayerTs...>>(s, prepend, threadContext);
			}

			//define forward function
			template<typename LayerT> requires std::is_base_of_v<Subsequence<Layer0, LayerTs...>, LayerT>
			static void forward(LayerT& layer, Threads::ThreadContext& ctx) {
				Layer0::forward(layer.head, ctx);
				Subsequence<LayerTs...>::forward(layer.tail, ctx);

			}

			template<typename LayerT> requires std::is_base_of_v<Subsequence<Layer0,LayerTs...>,LayerT>
			static void backward(LayerT& layer, Threads::ThreadContext& ctx) {
				Subsequence<LayerTs...>::backward(layer.tail, ctx);
				Layer0::backward(layer.head, ctx);
				

			}

			template<typename LayerT> requires std::is_base_of_v<Subsequence<Layer0,LayerTs...>,LayerT>
			static void modifyWeights(LayerT& layer, Threads::ThreadContext& ctx) {
				Subsequence<LayerTs...>::modifyWeights(layer.tail, ctx);
				Layer0::modifyWeights(layer.head, ctx);
			}

			static void saveAsBytes(Subsequence<Layer0, LayerTs...>& layer, std::ofstream& file, Threads::ThreadContext& ctx) {
				Layer0::saveAsBytes(layer.head, file, ctx);
				Subsequence<LayerTs...>::saveAsBytes(layer.tail, file, ctx);
			}

			static void readFromBytes(Subsequence<Layer0, LayerTs...>& layer, std::ifstream& file, Threads::ThreadContext& ctx) {
				Layer0::readFromBytes(layer.head, file, ctx);
				Subsequence<LayerTs...>::readFromBytes(layer.tail, file, ctx);
			}

		};

		//base case
		template<Layers::Layer Layer0>
		class Subsequence<Layer0> : public Layers::Interfaces::ILayer<
			typename Traits::getLayerTraits<Layer0>::InputDtype,
			typename Traits::getLayerTraits<Layer0>::InputShape,
			typename Traits::getLayerTraits<Layer0>::InputDevice,
			typename Traits::getLayerTraits<
			Layer0
			>::OutputDtype,
			typename Traits::getLayerTraits<
			Layer0
			>::OutputShape,
			typename Traits::getLayerTraits<
			Layer0
			>::OutputDevice,
			Traits::getLayerTraits<Layer0>::nThreads
		> {

		private:
			//alias the traits
			using InputDtype = typename Traits::getLayerTraits<Layer0>::InputDtype;
			using InputShape = typename Traits::getLayerTraits<Layer0>::InputShape;
			using InputDevice = typename Traits::getLayerTraits<Layer0>::InputDevice;

			using InputType = typename Traits::getLayerTraits<Layer0>::InputType;

			using OutputDtype = typename Traits::getLayerTraits<Layer0>::OutputDtype;
			using OutputShape = typename Traits::getLayerTraits<Layer0>::OutputShape;
			using OutputDevice = typename Traits::getLayerTraits<Layer0>::OutputDevice;

			using OutputType = typename Traits::getLayerTraits<Layer0>::OutputType;

			static constexpr const unsigned int nThreads = Traits::getLayerTraits<Layer0>::nThreads;

		public:
			Layer0 head;
			Subsequence(
				Tensors::Tensor<InputDtype, InputShape, nThreads, InputDevice> inputs
				, Tensors::Tensor<OutputDtype, OutputShape, nThreads, OutputDevice> outputs
			) : head(inputs, outputs) {}

			Subsequence(Subsequence<Layer0>&) = delete; //delete copy constructor
			Subsequence(Subsequence<Layer0>&&) = default; //keep move constructor



			void initializeWeights(Threads::ThreadContext& ctx) {
				head.initializeWeights(ctx);
			}

			void setWeights(uint8_t*& handle, Threads::ThreadContext& threadContext) {
				head.setWeights(handle, threadContext);
			}

			template<typename LayerT>
			void displayWeights(std::stringstream& s, std::string prepend, Threads::ThreadContext& threadContext) const {
				head.template displayWeights<Layer0>(s, prepend + "\t", threadContext);
			}

			//define forward function
			static void forward(Subsequence<Layer0>& layer, Threads::ThreadContext& ctx) {
				Layer0::forward(layer.head, ctx);
			}

			static void backward(Subsequence<Layer0>& layer, Threads::ThreadContext& ctx) {
				Layer0::backward(layer.head, ctx);

			}

			static void modifyWeights(Subsequence<Layer0>& layer, Threads::ThreadContext& ctx) {
				Layer0::modifyWeights(layer.head, ctx);
			}

			static void saveAsBytes(Subsequence<Layer0>& layer, std::ofstream& file, Threads::ThreadContext& ctx) {
				Layer0::saveAsBytes(layer.head, file, ctx);
			}

			static void readFromBytes(Subsequence<Layer0>& layer, std::ifstream& file, Threads::ThreadContext& ctx) {
				Layer0::readFromBytes(layer.head, file, ctx);
			}
		};
	}
}