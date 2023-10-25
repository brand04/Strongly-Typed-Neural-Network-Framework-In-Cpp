#pragma once
#include "../../helpers/fixed_string.h"
#include "../../helpers/string_collection.h"
#include "../../threads/thread_context.h"
#include "../operations/display.h"
#include "../../concepts/is_string_collection.h"
#include <stdio.h>
namespace NN {
	namespace Layers {
		namespace Phantom {

			/// <summary>
			/// Outputs to stdout a trace of the operations occuring on the wrapped layer
			/// </summary>
			/// <typeparam name="Layer"></typeparam>
			template<typename Layer>
			struct Tracer : public Layer {

			public:
				static constexpr StringCollection name = StringCollection("Traced ", Layer::name);

				template<typename LayerT> requires std::is_base_of_v<Tracer<Layer>,LayerT>
				inline static void forward(LayerT& layer, Threads::ThreadContext& ctx) {
					if constexpr (Concepts::IsStringCollection<decltype(LayerT::name)>) {
						FixedString tmp = LayerT::name.fix();
						std::cout << tmp.asStringView() << " : forward\n";
					}
					else {
						std::cout << Layer::name.asStringView() << " : forward\n";
					}
					Layer::forward(layer,ctx); //delegate to the forward function of Layer rather than this tracing forward function
				}
				
				template<typename LayerT> requires std::is_base_of_v<Tracer<Layer>, LayerT>
				inline static void backward(LayerT& layer, Threads::ThreadContext& ctx) {
					if constexpr (Concepts::IsStringCollection<decltype(LayerT::name)>) {
						FixedString tmp = LayerT::name.fix();
						std::cout << tmp.asStringView() << " : backward\n";
					}
					else {
						std::cout << LayerT::name.asStringView() << " : backward\n";
					}
					Layer::backward(layer,ctx);
				}
				
				template<typename LayerT> requires std::is_base_of_v<Tracer<Layer>, LayerT>
				inline static void modifyWeights(LayerT& layer, Threads::ThreadContext& ctx) {
					if constexpr (Concepts::IsStringCollection<decltype(LayerT::name)>) {
						FixedString tmp = LayerT::name.fix();
						std::cout << tmp.asStringView() << " : modify weights\n";
					}
					else {
						std::cout << LayerT::name.asStringView() << " : modify weights\n";
					}
					Layer::modifyWeights(layer,ctx);
				}

				using Layer::Layer;

			};
		}
	}
}