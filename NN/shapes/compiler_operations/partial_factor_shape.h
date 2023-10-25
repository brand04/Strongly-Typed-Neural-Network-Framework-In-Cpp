#pragma once
#include "../runtime_shape.h"
namespace NN {
	namespace Shapes {
		namespace Operations {
			/*
			Compute partial factors - dropping highest order axis

			if the input shape is N dimensional, this will result in an N-1 dimensional PartialFactorShape except if N=1, in which case a 1 dimensional PartialFactorShape


			For explanation, Shape<1,2,3,4> will be the Shape 4x3x2x1 from lowest to highest axis (ie 4 is the lowest axis)
			ie Shape<1,2,3,4> -> partialFactorShape<24,12,4>
			to unflatten 22, compute
			22/24 -> 0
			22/12 -> 1
			(22-1*12)/4 = 22%12/4 = 10/4 -> 2
			(10-2*4) = 22%12%4 = 10%4 = 2
			giving Shape<0,1,2,2>
			This is the mapping of the number 22 to this shape-space

			Used for precomputing flatten/unflatten operations when done as a runtime operation

			*/


			template<unsigned int P0, unsigned int ...PS>
			struct PartialFactorShape {
				static constexpr unsigned int asArray[1 + sizeof...(PS)] = { P0,PS... };
				static __device__ __host__ constexpr const RuntimeShape<1 + sizeof...(PS)> asRuntimeShape() {
					return RuntimeShape{ {P0,PS...} };
				}
			};

			struct EmptyPartialFactorShape {};
		}
	}
}