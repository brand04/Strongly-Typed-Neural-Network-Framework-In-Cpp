#pragma once
#include "../forward_declaration.h"
#include "./max_dimension.h"
namespace NN {

	namespace Shapes {

		template<typename Shape, unsigned int padValue, unsigned int paddingLength>
		struct PadShape;

		template<unsigned int  ... components, unsigned int padValue, unsigned int paddingLength> requires (paddingLength >0)
		struct PadShape<Shape<components...>, padValue, paddingLength> {
			using type = PadShape<Shape<padValue, components...>, padValue, paddingLength-1>::type;

		};

		template<typename Shape, unsigned int padValue>
		struct PadShape<Shape, padValue, 0> {
			using type = Shape;
		};


		template<typename ShapeReference, typename Shape, unsigned int padValue = 1>
		using PadTo = typename PadShape<Shape, padValue, ShapeReference::dimension - Shape::dimension>::type;

		template<typename Shape0, typename ... Shapes>
		using PadToMax = typename PadShape < Shape0, 1, getMaxDimension<Shape0, Shapes...> - Shape0::dimension>::type;
	}


}