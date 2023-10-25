#pragma once
#include "../overlay/biased_overlay.h"
#include "../../asserts/dimensions.h"
#include "../../helpers/string_literal.h"
#include "../../shapes/includes.h"

namespace NN {
	namespace Layers {

		//redefine the interface as receiving and giving out different types to the Overlay we are using the implementation of, a good example of a shape-redefining layer, as it declares its shapes to be different from the shapes used by the implementation
		//TODO: specialized Linear kernel for faster matrix multiplication


		//Takes and returns a vector of values that are multiplied with a matrix and have a biases added
		template<typename Dtype, typename Device, typename InputShape, typename OutputShape, unsigned int threads, typename WeightAdjuster = NN::WeightModifiers::Linear<typename Dtype::Type>>
		class BiasedLinear :
			//A Biased Linear Layer is a special case of Biased Overlay, where a n x 1 tensor is overlayed on a n by m matrix producing a 1 by m output - though a more specialized kernel would be more optimal
			public BiasedOverlay<Dtype, Device, typename InputShape::raise, Shapes::transpose<typename OutputShape::raise>, threads, WeightAdjuster>

			
			
		{
		private:
			
			static_assert(struct_assert<Asserts::BulkAssert<Asserts::AssertIsOneDimensional, InputShape, OutputShape>>, "Failed to create Biased Linear Layer");
			using Super = BiasedOverlay<Dtype, Device, typename InputShape::raise, Shapes::transpose<typename OutputShape::raise>, threads, WeightAdjuster>;
		public:
			using LayerTraits = Traits::LayerTraits<Dtype, InputShape, Device, Dtype, OutputShape, Device, threads>;
			static constexpr const StringLiteral name = StringLiteral("Biased Linear");
			

			BiasedLinear(Tensors::Tensor<Dtype, InputShape, threads, Device> inputs, Tensors::Tensor<Dtype, OutputShape, threads, Device> outputs)
				: Super(inputs.raise(), outputs.raise().transpose()) {} //perform tensor transformations so that the overlay layer receives 2D


			
		};
	}
}