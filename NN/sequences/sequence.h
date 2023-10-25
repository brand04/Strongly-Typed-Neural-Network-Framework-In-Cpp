#pragma once
#include "./subsequence.h"
#include "../helpers/string_t.h"
#include "../helpers/string_collection.h"

#include "../asserts/sequence_asserts.h"


namespace NN {
	namespace Sequences {
		//A complete sequence of layers - no traits (no append/prepend/ect operations)
		template<Layers::Layer Layer0, Layers::Layer ... Layers>
		class SequenceImpl<Layer0, Layers...> : public Sequences::Subsequence<Layer0, Layers...>{
		private:
			static_assert(struct_assert<Asserts::AssertSequence<Layer0, Layers...>>, "Attempted to create a sequence, but it was malformed");
		public:
			static constexpr const StringLiteral name = StringLiteral("Layer Sequence");

			
			
			//note that we use a FixedString here rather than a StringCollection - This is because the tree-like datastructure of StringCollection doesnt work here due to the fact that we are attempting to reference local-lifetimed StringCollections (in our postprocessing/fold of sub-layer's fullnames)
			//To fix would requre inserting multiple construction parameters from a single element of a pack, which idk if that is currently possible
			template<typename LayerT, typename prepend = StringT<>, typename prepender = StringT<'\t'>>
			static constexpr const FixedString fullname = StringCollection(prepend::string, LayerT::name, " (", Traits::getLayerTraits<LayerT>::InputShape::string,
				" [ ", Traits::getLayerTraits<LayerT>::InputDevice::string, " ] -> ",
				Traits::getLayerTraits<LayerT>::OutputShape::string, " [ ",
				Traits::getLayerTraits<LayerT>::OutputDevice::string, " ] ) {\n",
				StringCollection(Layer0::template fullname<Layer0, typename prepend::append<prepender>, prepender>, "\n") , StringCollection(Layers::template fullname<Layers, typename prepend::append<prepender>, prepender>, "\n") ... , prepend::string, "}").fix();

			template<typename LayerT>
			void displayWeights(std::stringstream& s, std::string prepend, Threads::ThreadContext& threadContext) const {
				s << prepend << "Sequence (\n";
				Sequences::Subsequence<Layer0, Layers...>::template displayWeights<Sequences::Subsequence<Layer0, Layers...>>(s, prepend + "\t", threadContext);
				s << prepend << ")\n";
			}



			using Sequences::Subsequence<Layer0, Layers...>::Subsequence;
		};
	}

	template<Layers::Layer Layer0, Layers::Layer... Layers>
	class Sequence<Layer0, Layers...> : public Sequences::SequenceImpl<Layer0, Layers...>, public Traits::SequenceTraits<Sequence, Layer0, Layers...> {
	public:
		using Sequences::SequenceImpl<Layer0, Layers...>::SequenceImpl; 
	};

}