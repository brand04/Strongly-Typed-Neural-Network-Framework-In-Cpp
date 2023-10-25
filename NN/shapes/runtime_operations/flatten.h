#pragma once
#include "../compiler_operations/partial_factor_shape.h"
#include "../runtime_shape.h"
#include <cuda_runtime.h>

namespace NN {
	namespace Shapes {
		namespace Operations {
			template<const unsigned int dimension, typename PartialFactorT>
			struct runtimeFlattener;

			template<const unsigned int dimension, const unsigned int Factor0>
			struct runtimeFlattener<dimension, PartialFactorShape<Factor0>> {
				__host__ __device__ const static inline unsigned int flatten(RuntimeShape<dimension> subShape, unsigned int i) {
					return (subShape.components[i] * Factor0) + subShape.components[i + 1];
				}
			};

			//special case, dimension = 1, and thus no partial factors
			template<const unsigned int dimension>
			struct runtimeFlattener<dimension, EmptyPartialFactorShape> {
				__host__ __device__ const static inline unsigned int flatten(RuntimeShape<dimension> subShape, unsigned int i) {
					return subShape.components[i];
				}
			};

			template<const unsigned int dimension, const unsigned int Factor0, const unsigned int ...Factors>
			struct runtimeFlattener < dimension, PartialFactorShape<Factor0, Factors...>> {
				__host__ __device__ const static inline unsigned int flatten(RuntimeShape<dimension> subShape, unsigned int i) {
					return (subShape.components[i] * Factor0) + runtimeFlattener<dimension, PartialFactorShape<Factors...>>::flatten(subShape, i + 1);
				}
			};
		}
	}
}