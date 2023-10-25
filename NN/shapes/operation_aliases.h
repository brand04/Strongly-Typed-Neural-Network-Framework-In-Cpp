#pragma once
#include "./compiler_operations/transpose.h"
#include "./compiler_operations/fill.h"
#include "shape_concept.h"

namespace NN {
	namespace Shapes {




		template<const unsigned int fillValue, const unsigned int Dimension>
		using fill = Operations::filler<void, fillValue, Dimension>::type;

		template<const unsigned int Dimension>
		using unit = Operations::filler<void, 1, Dimension>::type;

		template<const unsigned int Dimension>
		using zero = Operations::filler<void, 0, Dimension>::type;

		template<typename ShapeT> requires IsShape<ShapeT>
		using transpose = Operations::transposeShape<ShapeT>::type;
		

			
	}
}