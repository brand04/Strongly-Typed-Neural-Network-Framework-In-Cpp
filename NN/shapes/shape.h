#pragma once
#include <cuda_runtime.h>
#include "./compiler_operations/add.h"
#include "./compiler_operations/subtract.h"
#include "./compiler_operations/expand.h"
#include "./compiler_operations/shape_to_partial_factor_shape.h"
#include "./compiler_operations/partial_factor_shape.h"
#include "./compiler_operations/reduce.h"
#include "./compiler_operations/expand.h"
#include "./compiler_operations/flatten.h"
#include "./compiler_operations/unflatten.h"
#include "./compiler_operations/volume.h"
#include "./compiler_operations/reverse.h"

#include "./compiler_operations/shape_to_string_t.h"

#include "runtime_shape.h"

#include "./runtime_operations/flatten.h"
#include "./runtime_operations/unflatten.h"

#include "../helpers/string_t.h"


namespace NN {
	namespace Shapes {
		/**
		* Shape type
		*
		* Used to specify the dimensions and size of each dimension within a tensor
		* Contains various options to perform operations
		* 
		*
		* ::dimension - the number of axi this shape has
		* ::volume - the total space occupied by this shape - used for memory allocation
		* 
		* ::head - the value in the highest order dimension
		*
		* ::reduce - removes the highest order dimension from this shape - this should be handled by using multiple operations and combining
		* ::expand<i> - adds a highest order dimension - used to specify that a computation is run multiple times and different outputs produced
		* ::raise - expand<1>
		* 
		* ::add<shape> - sums along each dimension
		* ::subtract<shape> - subtracts along each dimension
		* 
		* ::flatten<shape> - returns a number representing the position of 'shape' within this shape
		* ::unflatten<n> - produces a shape representing the values along each dimension of the position n (inverse of flatten)
		*
		* ::asArray - gets the static constexpr array of the axi - HIGHEST ORDER FIRST
		* ::partialFactors - gets the partialFactorShape
		* ::partialFactors::asArray - gets the static constexpr array of the partial factors ([5,4,3,2,1] -> [24, 6, 2, 1])
		* 
		* ::asStringT - gets a StringT describing this shape
		* ::string - gets a reference to a static char[] containing a description of this shape (for example 5x4x9)
		*/

		template<const unsigned int C0, const unsigned int ...CS> struct Shape {

			//get the dimension of this shape (number of components)
			static constexpr unsigned int dimension = sizeof...(CS) + 1;
			//get the volume of this shape
			static constexpr unsigned int volume = Operations::getVolume<Shape<C0, CS...>, 1>::value;

			//get the highest dimension component
			static constexpr unsigned int head = C0; //reduce allows access to tail, provide access to the head

			//reduce the dimension of the shape by discarding the highest dimension component
			using reduce = Operations::reducer<Shape<C0, CS...>>::type;

			//raise the dimension of a shape by prepending an arbitary value
			template<const unsigned int CN>
			using expand = typename Operations::expander<Shape<C0, CS...>, CN>::type;

			//raise the dimension of the shape by prepending a 1
			using raise = Operations::expander<Shape<C0, CS...>, 1>::type;

			//get this shape as an array (useful for dynamic compuations) - only defined for host code - use asRuntimeShape().components[...] for cuda code
			static constexpr const unsigned int asArray[1 + sizeof...(CS)] = { C0,CS... };
			
			//TODO: double check that consteval fails on CPU due to nvcc interferring with constexpr-ness - same for partial factor shape ::asRuntimeShape
			__device__ __host__  static constexpr const RuntimeShape<dimension> asRuntimeShape() { 
				return RuntimeShape{ { C0, CS... } };
			}


			//get the partial factors - an N-1 dimensional shape containing the cumulative product of this shape, starting at the lowest order dimension
			using partialFactors = Operations::shapeToPartialFactorShape<Operations::reverse<Shape<C0, CS...>>, void>::type;

			//Run a bijection from subshapes -> {1,2,...,volume}, which provides a unique identifying number to that subshape
			template<typename SubShape>
			static constexpr unsigned int flatten = Operations::flattener<partialFactors, SubShape, 0>::value;

			//run a bijection from {1,2,...,volume} -> subshapes, which converts a unique subshape identifier, to that subshape
			template<const unsigned int flattened>
			using unflatten = Operations::unflattener<partialFactors, flattened, void>::type;

			//add two shapes together
			template<typename OtherShape> requires(IsShape<OtherShape>)
			using add = Operations::adder<Shape<C0, CS...>, OtherShape, void>::type;

			//subtract one shape from another
			template<typename OtherShape> requires (IsShape<OtherShape>)
			using subtract = Operations::subtracter<Shape<C0, CS...>, OtherShape, void>::type;

			//unflatten a runtime value, returing a RuntimeShape
			__host__ __device__ static const inline RuntimeShape<dimension> runtimeUnflatten(const unsigned int flattened) {
				return Operations::runtimeUnflattener<dimension, partialFactors>::unflatten(flattened);
			}


			//flattens a runtime shape, using some precomputed values
			__host__ __device__ static const inline unsigned int runtimeFlatten(RuntimeShape<dimension> subShape) {
				return Operations::runtimeFlattener<dimension, partialFactors>::flatten(subShape, 0);
			}

			//the shape as a StringT Type
			using asStringT = Operations::shapeToStringT < Shape<C0, CS...>>::type;

			//A reference to a compile-time string of this shape
			static constexpr const char (&string)[asStringT::length+1] = Operations::shapeToStringT < Shape<C0, CS...>>::type::value;
			
			

		};
	}
}