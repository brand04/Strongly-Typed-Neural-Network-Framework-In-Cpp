#pragma once
#include <tuple>
#include "../../layers/interfaces/i_layer.h"
#include "../../layers/abstract/layer_box.h"
#include "../../storage/unsafe/forward_declaration.h"
#include "../asserts/layer_asserts.h"
#include "../../devices/includes.h"
#include <cuda_runtime.h>
#include "../../tensors/tensors.h"
#include "../../helpers/tuple.h"
#include "../../threads/thread_context.h"

#include "../../kernels/elemental_reducer.cuh"
#include "../../kernels/copy_expansion.cuh"
#include <exception>
#include <thread>
#include "../../helpers/string_collection.h"
namespace NN {
	namespace Sequences {

	
		template<typename Reducer, bool doSpawnThreads, typename Layer0, typename ... LayerTs>
		class ConcurrentSequenceImpl : public Layers::Abstract::LayerBox<
			typename Traits::getLayerTraits<Layer0>::InputDtype,
			typename Traits::getLayerTraits<Layer0>::InputShape,
			typename Traits::getLayerTraits<Layer0>::InputDevice,
			typename Traits::getLayerTraits<Layer0>::OutputDtype,
			typename Traits::getLayerTraits<Layer0>::OutputShape,
			typename Traits::getLayerTraits<Layer0>::OutputDevice,
			Traits::getLayerTraits<Layer0>::nThreads > {
		protected:
			


			using InputShape = typename Traits::getLayerTraits<Layer0>::InputShape;
			using OutputShape = typename Traits::getLayerTraits<Layer0>::OutputShape;

			using InputDtype = typename Traits::getLayerTraits<Layer0>::InputDtype;
			using OutputDtype = typename Traits::getLayerTraits<Layer0>::OutputDtype;

			using InputType = typename Traits::getLayerTraits<Layer0>::InputType;
			using OutputType = typename Traits::getLayerTraits<Layer0>::OutputType;

			using InputDevice = typename Traits::getLayerTraits<Layer0>::InputDevice;
			using OutputDevice = typename Traits::getLayerTraits<Layer0>::OutputDevice;

			static constexpr const unsigned int nThreads = Traits::getLayerTraits<Layer0>::nThreads;

			using Self = ConcurrentSequenceImpl<Reducer, doSpawnThreads, Layer0, LayerTs...>;

			//asserts for the seqeuence to work

			static_assert((true & ... & std::same_as<InputShape, typename Traits::getLayerTraits<LayerTs>::InputShape>), "Expected all Concurrent layers to have the same Input Shape");
			static_assert((true & ... & std::same_as<OutputShape, typename Traits::getLayerTraits<LayerTs>::OutputShape>), "Expected all Concurrent layers to have the same Output Shape");
			static_assert((true & ... & std::same_as<InputDtype, typename Traits::getLayerTraits<LayerTs>::InputDtype>), "Expected all Concurrent layers to have the same Input Datatype");
			static_assert((true & ... & std::same_as<OutputDtype, typename Traits::getLayerTraits<LayerTs>::OutputDtype>), "Expected all Concurrent layers to have the same Output Datatype");
			static_assert((true & ... & std::same_as<InputDevice, typename Traits::getLayerTraits<LayerTs>::InputDevice>), "Expected all Concurrent layers to have the same Input Device");
			static_assert((true & ... & std::same_as<OutputDevice, typename Traits::getLayerTraits<LayerTs>::OutputDevice>), "Expected all Concurrent layers to have the same Output Device");

			static_assert(struct_assert<Asserts::BulkAssert<Asserts::AssertLayer, Layer0, LayerTs...>>, "Failed to create Concurrent sequence - the arguments supplied were not layers");
			static constexpr size_t parallels = sizeof...(LayerTs) + 1; //+1 for Layer0
			static constexpr size_t newBranches = (parallels-1) * nThreads; //number of new threads spawned
			static constexpr size_t branches = newBranches == 0 ? 1 : newBranches; //special case for parallels = 1 -> newBranches = 0 as T[0] is incomplete type


			Threads::ThreadContext ctxs[branches]; 
		
			std::exception* exceptions[branches];
			using OutputType = typename Traits::getLayerTraits<Layer0>::OutputType;

			Helpers::Tuple < Storage::TrainingStore<OutputDtype, OutputShape, nThreads, OutputDevice>, Storage::TrainingStore<OutputDtype, typename Traits::getLayerTraits<LayerTs>::OutputShape, nThreads, OutputDevice>...> outputStores;
			Helpers::Tuple<Storage::TrainingStore<InputDtype, InputShape, nThreads, InputDevice>, Storage::TrainingStore<InputDtype, typename Traits::getLayerTraits<LayerTs>::InputShape, nThreads, InputDevice>...> inputStores;
			Helpers::Tuple<Layer0, LayerTs...>  layers;
			



		private:
			template<size_t ... indicies> //indices = std::make_index_sequence<sizeof...(LayerTs)>()
			ConcurrentSequenceImpl(Tensors::Tensor<InputDtype, InputShape, nThreads, InputDevice> inputs, Tensors::Tensor<OutputDtype, OutputShape, nThreads, OutputDevice> outputs, std::index_sequence<indicies...> seq) :
				Layers::Abstract::LayerBox<InputDtype, InputShape, InputDevice, OutputDtype, OutputShape, OutputDevice, nThreads>(inputs, outputs), 
				outputStores(),
				inputStores(),
				layers(
					Helpers::ArgPack{std::tuple(Helpers::get<0>(inputStores).makeTensor(), Helpers::get<0>(outputStores).makeTensor()) },
					Helpers::ArgPack{std::tuple(Helpers::get<indicies+1>(inputStores).makeTensor(), Helpers::get<indicies + 1>(outputStores).makeTensor()) }...
				 ),
				ctxs(),
				exceptions() {
				if constexpr (newBranches > 0) {
					for (unsigned int i = 0; i < newBranches; i++) {
						ctxs[i].threadId = (i / (parallels-1));
						if constexpr (Devices::CUDADevice<InputDevice>) {
							ctxs[i].prepareCuda();
						}
						exceptions[i] = nullptr;
					}
				}
			}
		
			//case no indicies
			template<> //indices = std::make_index_sequence<sizeof...(LayerTs)>()
			ConcurrentSequenceImpl(Tensors::Tensor<InputDtype, InputShape, nThreads, InputDevice> inputs, Tensors::Tensor<OutputDtype, OutputShape, nThreads, OutputDevice> outputs, std::index_sequence<> seq) :
				Layers::Abstract::LayerBox<InputDtype, InputShape, InputDevice, OutputDtype, OutputShape, OutputDevice, nThreads>(inputs,outputs),
				inputStores(),
				outputStores(),
				layers(
					Helpers::ArgPack{std::tuple(Helpers::get<0>(inputStores).makeTensor(), Helpers::get<0>(outputStores).makeTensor()) }
				),
				exceptions(),
				ctxs() {
				if constexpr (newBranches > 0) {
					for (unsigned int i = 0; i < newBranches; i++) {
						ctxs[i].threadId = i;
						if constexpr (Devices::CUDADevice<InputDevice>) {
							ctxs[i].prepareCuda(); //prepare streams for these threads
						}
						exceptions[i] = nullptr;
					}
				}
				
			}
			public:
				
				static constexpr StringLiteral name = "Concurrent Sequence";
				template<typename LayerT, typename prepend = StringT<>, typename prepender = StringT<'\t'>>
				static constexpr const FixedString fullname = StringCollection(
					prepend::string, LayerT::name, " (", Traits::getLayerTraits<LayerT>::InputShape::string, //name, features
					" [ ", Traits::getLayerTraits<LayerT>::InputDevice::string, " ] -> ", //more features
					Traits::getLayerTraits<LayerT>::OutputShape::string, " [ ", //more features
					Traits::getLayerTraits<LayerT>::OutputDevice::string, " ] )", 
					" - Reduced with ", Reducer::name, " {\n",  //reducer function and opening brace
					StringCollection(Layer0::template fullname<Layer0, typename prepend::append<prepender>, prepender>, "\n"), //first layer
					StringCollection(LayerTs::template fullname<LayerTs, typename prepend::append<prepender>, prepender>, "\n") ...,  //other layers
					prepend::string, "}" //closing brace
				).fix();
					
			private:

			//in order to initialize std::index_sequences we must use an additional layer of indirection, so the functions are all private here and then we have public ones that call the private ones with the correct std::index_sequence

			template<size_t ... indicies>
			inline void threadForward(
				References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
				References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,	
				Threads::ThreadContext& ctx,
				std::index_sequence<0,indicies...>
			) {
				//copy
				Kernels::CopyExpansion::copyExpansionForward<InputType, InputDevice, InputShape, InputShape, typename Traits::getLayerTraits<LayerTs>::InputShape...>(inputs, ctx, Helpers::get<0>(inputStores).data.asMutable(ctx), Helpers::get<indicies>(inputStores).data.asMutable(ctx)...);
				ctx.synchronize();


				if constexpr (doSpawnThreads) {
					std::thread threads[parallels - 1] = {
						std::thread([&layer = Helpers::get<indicies>(layers), &threadContext = ctxs[(indicies - 1) + (ctx.threadId * (parallels - 1))], &exceptionPtr = exceptions[ctx.threadId * (indicies - 1)]]() -> void {
							try {
								exceptionPtr = nullptr;
								layer.forward(layer, threadContext);
								threadContext.synchronize();
							}
							catch (const std::exception& e) {

								*exceptionPtr = e;
							}


						}) ...
					};

					//run on the current thread
					try {

						Helpers::get<0>(layers).forward(Helpers::get<0>(layers), ctx);
					}
					catch (const std::exception& e) {

						std::cerr << "An exception occured during concurrent sequence thread 0 : " << e.what() << "\n";
					}

					//wait for threads to complete
					for (size_t i = 0; i < parallels - 1; i++) {

						threads[i].join();
						if (exceptions[i] != nullptr) {
							std::cerr << "An exception occured during concurrent sequence thread " << std::to_string(i) << " : " << exceptions[i]->what() << "\n";
							throw (exceptions[i]);
						}
					}
				}
				else {
					Helpers::get<0>(layers).forward(Helpers::get<0>(layers), ctx);
					((Helpers::get<indicies>(layers).forward(Helpers::get<indicies>(layers), ctxs[(indicies-1)+(ctx.threadId*(parallels-1))])), ...); 
					//the math for ctxs is that we generate parallel-1 ctx branches for each of the nThreads (and use the current context for the rest)
					//we store the ctxs by consequetive thread ids (each of length parallel-1), so index using essentially 2 dimensional indexing style
					
					//synchronize
					ctx.synchronize();
					(ctxs[ctx.threadId * (indicies - 1)].synchronize(), ...);
				}
				
				
				Kernels::Reducers::elementalReduceForward<Reducer, OutputDtype, OutputDevice, OutputShape, OutputShape, typename Traits::getLayerTraits<LayerTs>::OutputShape...>(outputs, ctx, Helpers::get<0>(outputStores).data.asImmutable(ctx),Helpers::get<indicies>(outputStores).data.asImmutable(ctx)...);
			}

			template<size_t ... indicies>
			inline void threadBackward(
				References::ImmutableShapedDeviceReference<InputType,InputShape, InputDevice> inputs,
				References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
				References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
				References::MutableShapedDeviceReference<InputType, InputShape, InputDevice> preNodeDeltas,
				Threads::ThreadContext& ctx,
				std::index_sequence<0,indicies...> seq
			) {

				Kernels::Reducers::elementalReduceBackward<Reducer, OutputDtype, OutputDevice, OutputShape, OutputShape, typename Traits::getLayerTraits<LayerTs>::OutputShape...>(postNodeDeltas, outputs, ctx, Helpers::get<0>(outputStores).data.asImmutable(ctx), Helpers::get<indicies>(outputStores).data.asImmutable(ctx)..., Helpers::get<0>(outputStores).deltas.asMutable(ctx), Helpers::get<indicies>(outputStores).deltas.asMutable(ctx)...);

				ctx.synchronize();
				if constexpr (doSpawnThreads) {
					std::thread threads[parallels - 1] = {
						std::thread([&layer = Helpers::get<indicies>(layers), &threadContext = ctxs[(indicies - 1) + (ctx.threadId * (parallels - 1))], &exceptionPtr = exceptions[ctx.threadId * (indicies - 1)]]() -> void {
							try {
								exceptionPtr = nullptr;
								layer.backward(layer, threadContext);
							}
							catch (const std::exception& e) {

								*exceptionPtr = e;
							}
						}) ...
					};

					//run on the current thread
					try {

						Helpers::get<0>(layers).backward(Helpers::get<0>(layers), ctx);
					}
					catch (const std::exception& e) {

						std::cerr << "An exception occured during concurrent sequence thread 0 : " << e.what() << "\n";
					}

					//wait for threads to complete
					for (size_t i = 0; i < parallels - 1; i++) {

						threads[i].join();
						if (exceptions[i] != nullptr) {
							std::cerr << "An exception occured during concurrent sequence thread " << std::to_string(i) << " : " << exceptions[i]->what() << "\n";
							throw (exceptions[i]);
						}
					}
				}
				else {
					Helpers::get<0>(layers).backward(Helpers::get<0>(layers), ctx);
					((Helpers::get<indicies>(layers).backward(Helpers::get<indicies>(layers), ctxs[(indicies - 1) + (ctx.threadId * (parallels - 1))])), ...);
				}

				Kernels::CopyExpansion::copyExpansionBackward<InputDtype, InputDevice, InputShape, InputShape, typename Traits::getLayerTraits<LayerTs>::InputShape...>(preNodeDeltas, ctx,  Helpers::get<0>(inputStores).deltas.asImmutable(ctx), Helpers::get<indicies>(inputStores).deltas.asImmutable(ctx)...);

				
			}

			template<size_t ... indicies>
			inline void threadModifyWeights(
				References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
				References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
				Threads::ThreadContext& ctx, std::index_sequence<0,indicies...> seq
			) {
				if constexpr (doSpawnThreads) {
					std::thread threads[parallels - 1] = {
						std::thread([&layer = Helpers::get<indicies>(layers), &threadContext = ctxs[(indicies - 1) + (ctx.threadId * (parallels - 1))], &exceptionPtr = exceptions[ctx.threadId * (indicies - 1)]]() -> void {
							try {
								exceptionPtr = nullptr;
								layer.modifyWeights(layer, threadContext);
							}
							catch (const std::exception& e) {

								*exceptionPtr = e;
							}
						}) ...
					};

					//run on the current thread
					try {

						Helpers::get<0>(layers).modifyWeights(Helpers::get<0>(layers), ctx);
					}
					catch (const std::exception& e) {

						std::cerr << "An exception occured during concurrent sequence thread 0 : " << e.what() << "\n";
						throw e;
					}

					//wait for threads to complete
					for (size_t i = 0; i < parallels - 1; i++) {

						threads[i].join();
						if (exceptions[i] != nullptr) {
							std::cerr << "An exception occured during concurrent sequence thread " << std::to_string(i) << " : " << exceptions[i]->what() << "\n";
							throw (exceptions[i]);
						}
					}
				}
				else {
					Helpers::get<0>(layers).modifyWeights(Helpers::get<0>(layers), ctx);
					((Helpers::get<indicies>(layers).modifyWeights(Helpers::get<indicies>(layers), ctxs[(indicies - 1) + (ctx.threadId * (parallels - 1))])), ...);
				}

			}

			template<size_t ... indicies>
			void displayWeights(std::stringstream& s, std::string prepend, Threads::ThreadContext& ctx, std::index_sequence<indicies...> seq) const {
				(Helpers::get<indicies>(layers).template displayWeights<decltype(Helpers::get<indicies>(layers))>(s, prepend + "\t", ctx), ...);
			}

			template<size_t ... indicies>
			void initializeWeights(Threads::ThreadContext& ctx, std::index_sequence<indicies...> seq) {
				(Helpers::get<indicies>(layers).initializeWeights(ctx), ...);
			}

			template<size_t ... indicies>
			void setWeights(uint8_t*& handle, Threads::ThreadContext& threadContext, std::index_sequence<indicies...> seq) {
				((Helpers::get<indicies>(layers).setWeights(handle, threadContext)), ...);
			}

			template<size_t ... indicies>
			static void saveAsBytes(Self& layer, std::ofstream& file, Threads::ThreadContext& ctx, std::index_sequence<indicies...> seq) {
				(std::tuple_element_t<indicies, std::tuple<Layer0, LayerTs...>>::saveAsBytes(Helpers::get<indicies>(layer.layers), file, ctx), ...); //figure out the type of the layer and run that type's save function with the correct layer (which will have that type)
			}
			template<size_t ... indicies>
			static void readFromBytes(Self& layer, std::ifstream& file, Threads::ThreadContext& ctx, std::index_sequence<indicies...> seq) {
				(std::tuple_element_t<indicies,std::tuple<Layer0,LayerTs...>>::readFromBytes(Helpers::get<indicies>(layer.layers), file, ctx), ...);
			}

		public:

			//indirection to private functions using correctly intialized index sequences
		
			inline void threadForward(
				References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
				References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
				Threads::ThreadContext& ctx
			) {
				threadForward(inputs, outputs, ctx, std::make_index_sequence<parallels>());
			}

			
			inline void threadBackward(References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
				References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
				References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
				References::MutableShapedDeviceReference<InputType, InputShape, InputDevice> preNodeDeltas,
				Threads::ThreadContext& threadContext
			) {
				threadBackward(inputs, outputs, postNodeDeltas, preNodeDeltas, threadContext, std::make_index_sequence<parallels>());
			}

			inline void threadModifyWeights(
				References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
				References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
				Threads::ThreadContext& ctx
			) {
				threadModifyWeights(inputs, postNodeDeltas, ctx, std::make_index_sequence<parallels>());
			}
			
			template<typename LayerT>
			void displayWeights(std::stringstream& s, std::string prepend, Threads::ThreadContext& threadContext) const {
				displayWeights(s, prepend, threadContext, std::make_index_sequence<parallels>());
			}


			void initializeWeights(Threads::ThreadContext& ctx) {
				initializeWeights(ctx, std::make_index_sequence<parallels>());
			}

			void setWeights(uint8_t*& handle, Threads::ThreadContext& threadContext) {
				setWeights(threadContext, std::make_index_sequence<parallels>());
			}

			static void saveAsBytes(Self& layer, std::ofstream& file, Threads::ThreadContext& ctx) {
				saveAsBytes(layer, file, ctx, std::make_index_sequence<parallels>());
			}

			static void readFromBytes(Self& layer, std::ifstream& file, Threads::ThreadContext& ctx) {
				readFromBytes(layer, file, ctx, std::make_index_sequence<parallels>());
			}
			ConcurrentSequenceImpl(Tensors::Tensor<InputDtype, InputShape, nThreads, InputDevice> inputs, Tensors::Tensor<OutputDtype, OutputShape, nThreads, OutputDevice> outputs) : ConcurrentSequenceImpl(inputs, outputs, std::make_index_sequence<sizeof...(LayerTs)>()) {
			}

			ConcurrentSequenceImpl(Self&) = delete; // delete copy constructor
			ConcurrentSequenceImpl(Self&&) = default;


		};

		template<typename Reducer, typename Layer0, typename ... Layers>
		using ConcurrentSequence = ConcurrentSequenceImpl<Reducer, false, Layer0, Layers...>; //by default dont spawn new threads
		
	}
}