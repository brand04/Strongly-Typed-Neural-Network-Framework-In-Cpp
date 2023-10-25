#pragma once
#include "../../traits/trait_get.h"
#include "../../devices/includes.h"
#include "../../references/includes.h"
#include "../../threads/thread_context.h"
#include "../../kernels/map.cuh"
#include "../../functions/map_functions/mappable.h"
#include "../../asserts/map_function_asserts.h"
namespace NN {
	namespace Layers {
		namespace Phantom {

			template<typename LayerT,  typename Operation>
				class ErrorPreprocessor: public LayerT {
				static_assert(struct_assert<Asserts::AssertMappableOn<Operation,typename Traits::getLayerTraits<LayerT>::OutputDtype>>, "Expected a function that can use the Dtype of the wrapped layer");
				using OutputType = typename Traits::getLayerTraits<LayerT>::OutputType;
				
				using InputType = typename Traits::getLayerTraits<LayerT>::InputType;
				using OutputDevice = typename Traits::getLayerTraits<LayerT>::OutputDevice;
				using InputDevice = typename Traits::getLayerTraits<LayerT>::InputDevice;
				using Dtype = typename Traits::getLayerTraits<LayerT>::OutputDtype;
				using OutputShape = typename Traits::getLayerTraits<LayerT>::OutputShape;
				using InputShape = typename Traits::getLayerTraits<LayerT>::InputShape;
				public:
					void inline threadBackward(
						References::ImmutableShapedDeviceReference<InputType, InputShape, InputDevice> inputs,
						References::ImmutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> outputs,
						References::MutableShapedDeviceReference<OutputType, OutputShape, OutputDevice> postNodeDeltas,
						References::MutableShapedDeviceReference<InputType, InputShape, InputDevice> preNodeDeltas,
						Threads::ThreadContext& ctx
					) {


						Kernels::Map::map<Operation, OutputShape, OutputType, OutputDevice>(postNodeDeltas, ctx);
						ctx.synchronize();
						//run wrapped layer with the modified error node deltas
						LayerT::template backward<LayerT>(*this, ctx);

							
					}

					using LayerT::LayerT;
			};
		}
	}
}