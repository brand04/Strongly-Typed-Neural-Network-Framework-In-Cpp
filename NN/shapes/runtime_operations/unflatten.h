#pragma once
#include "../runtime_shape.h"
#include "../compiler_operations/partial_factor_shape.h"
#include <cuda_runtime.h>

namespace NN {
	namespace Shapes {
		namespace Operations {

			template<const unsigned int dimension, typename PartialFactorT>
			struct runtimeUnflattener;

			template<const unsigned int dimension, const unsigned int ...Factors>
			struct runtimeUnflattener < dimension, PartialFactorShape<Factors...>> {
				static_assert(sizeof...(Factors) == dimension - 1, "Expected a partial factor shape with dimensionality of 1 less than the supplied dimension");
				__host__ __device__ const static RuntimeShape<dimension> unflatten(unsigned int flattened) {
					RuntimeShape<dimension> rshape{{Factors...}};

					unsigned int tmp;
					for (int i = 0; i < dimension - 1; i++) {
						tmp = rshape.components[i];
						rshape.components[i] = flattened / rshape.components[i];
						flattened = flattened % tmp;
					}
					rshape.components[dimension - 1] = flattened;
					return rshape;
				}
			};

			//special case - dimension = 1, therefore no partial factors
			template<>
			struct runtimeUnflattener < 0, EmptyPartialFactorShape> {
				__host__ __device__ const static RuntimeShape<0> unflatten(unsigned int flattened) {
					RuntimeShape<0> rshape;
					rshape.components[0] = flattened;
					return rshape;
				}
			};
		}
	}
}