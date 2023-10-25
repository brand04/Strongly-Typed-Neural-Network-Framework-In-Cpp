#pragma once
#include "../shapes/includes.h"
#include "../storage/forward_declarations.h"
#include "../devices/includes.h"
#include "../dtypes/concept.h"

namespace NN {
	namespace Tensors {
		template<Dtypes::Dtype Dtype, typename ShapeT, const unsigned int threads, Devices::Device Device > requires Shapes::IsShape<ShapeT> struct Tensor; 



		/// <summary>
		/// describes a flow of memory from one layer to the next, by comparing type, shape, threads, device at compile time, and a reference to the common-memory at runtime
		/// TODO: switch from using a Store ptr to something more flexible such as MutableShapedReferences, and also allow deltas to not be created
		/// </summary>
		
		template<Dtypes::Dtype Dtype, typename ShapeT, const unsigned int threads, Devices::Device Device> requires Shapes::IsShape<ShapeT>
		struct Tensor {
			
			Storage::TrainingStore<Dtype, ShapeT, threads, Device>* const link;
			Tensor(Storage::TrainingStore<Dtype, ShapeT, threads, Device>* const linkToData) : link(linkToData) {}

			template<typename OtherShape> requires Shapes::IsShape<OtherShape>
			explicit operator Tensor<Dtype, OtherShape, threads, Device>(){
				static_assert(OtherShape::volume == ShapeT::volume, "Attempted to cast to a tensor to another shape that did not have the same volume");
				return Tensor<Dtype, OtherShape, threads, Device>(
					reinterpret_cast<Storage::TrainingStore<Dtype, OtherShape, threads,Device>*const >(link) //reinterpret the store as a different store, with assurances from the static_assert that the volumes will be equal
				);
			}

			//prepends a 1 to the shape of this tensor
			Tensor<Dtype, typename ShapeT::raise, threads,Device> raise() {
				return static_cast<Tensor<Dtype, typename Shapes::Operations::expander<ShapeT,1>::type, threads, Device>>(*this);
			}

			
			//operation only available on some tensors - transposes a 2 dimensional tensor if 1 of the dimensions is of unit length (TODO: extend to arbitary dimensions)
			template<typename ValidShapeT = ShapeT> requires requires {typename Shapes::transpose<ValidShapeT>; }
			Tensor<Dtype, Shapes::transpose<ValidShapeT>, threads, Device> transpose() {
				return static_cast<Tensor<Dtype, Shapes::transpose<ValidShapeT>, threads, Device>>(*this);
			}

			

		};

		
		

		/*
		Type alias for a single dimensional tensor
		*/
		template<Dtypes::Dtype Dtype, const unsigned int length, const unsigned int threads, Devices::Device Device = Devices::CPU>
		using Vector = Tensor<Dtype, Shapes::Shape<length>, threads, Device>;




	}

}