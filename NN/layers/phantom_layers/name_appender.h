#pragma once
#include "../../asserts/layer_asserts.h"
#include "../../asserts/string_asserts.h"
namespace NN {
	namespace Layers {
		namespace Phantom {

			
			/// <summary>
			/// A Layer wrapper type whose sole purpose is to change the display name of the underlying layer, whilst keeping all of the functionality of the wrapped layer the same
			/// </summary>
			/// <typeparam name="LayerT"> The wrapped underlying layer type</typeparam>
			/// <typeparam name="AppendStringT">The StringT representing the string to append to the end of the name of the wrapped layer</typeparam>
			template<typename LayerT, typename AppendStringT>
			struct NameAppender : public  LayerT {
				//assert we have a valid Layer type and a valid StringT type
				static_assert(struct_assert<Asserts::AssertLayer<LayerT>>, "Failed to create NameAppender wrapper : the underlying layer is not a valid layer");
				static_assert(struct_assert<Asserts::AssertIsStringT<AppendStringT>>, "Failed to create NameAppender wrapper : the AppendStringT is not a valid StringT");

			public:
				static constexpr StringCollection name = StringCollection(LayerT::name , AppendStringT::value);

				using LayerT::LayerT;
			};
		}
	}

}