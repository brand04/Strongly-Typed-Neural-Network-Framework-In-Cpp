#pragma once

namespace NN {
	namespace Shapes {

		template<unsigned int Dimension> struct RuntimeShape;


		/*
		Runtime Shape - supports non-constexpr flattening and unflattening
		*/
		template<unsigned int Dimension>
		struct RuntimeShape {
			unsigned int components[Dimension];

			__host__ __device__ RuntimeShape<Dimension> add(const RuntimeShape<Dimension> other) const {
				RuntimeShape r;
				for (int i = 0; i < Dimension; i++) {
					r.components[i] = components[i] + other.components[i];
				}
				return r;
			}
		};
	}
}