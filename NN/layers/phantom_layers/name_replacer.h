#pragma once
#include "../../asserts/layer_asserts.h"
#include "../../asserts/string_asserts.h"
#include "../../helpers/fixed_string.h";
#include "../../helpers/string_collection.h";
#include "../../helpers/string_literal.h";

namespace NN {
	namespace Layers {
		namespace Phantom {


			/// <summary>
			/// A Layer wrapper type whose sole purpose is to change the display name of the underlying layer, whilst keeping all of the functionality of the wrapped layer the same
			/// </summary>
			/// <typeparam name="LayerT"> The wrapped underlying layer type</typeparam>
			/// <typeparam name="AppendStringT">The StringT representing the string to append to the end of the name of the wrapped layer</typeparam>
			template<typename LayerT, typename ReplaceStringT>
			struct NameReplacer : public LayerT { //CRTP
				//assert we have a valid Layer type and a valid StringT type
				static_assert(struct_assert<Asserts::AssertLayer<LayerT>>, "Failed to create NameReplacer wrapper : the underlying layer is not a valid layer");
				static_assert(struct_assert<Asserts::AssertIsStringT<ReplaceStringT>>, "Failed to create NameReplacer wrapper : the ReplaceStringT is not a valid StringT");

			public:
				static constexpr StringLiteral name = ReplaceStringT::value;

				using LayerT::LayerT;
			};
		}
	}

}