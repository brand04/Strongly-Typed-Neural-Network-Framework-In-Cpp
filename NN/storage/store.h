#pragma once
#include "./unsafe/includes.h"
#include "../shapes/shape_concept.h"
#include "../threads/thread_context.h"
#include "../dtypes/concept.h"
namespace NN {
	namespace Storage {

		//wraps an unsafe store with shape and thread information so that it can be type-checked

		//this means that the type is checked to be a declared data type, the shape of the store is used to determine it's size, and it is incompatible with other shaped stores, and the number threads must be declared as well as which device is being used

		template<Dtypes::Dtype Dtype, Shapes::IsShape Shape, const unsigned int threads, Devices::Device Device = Devices::CPU>
		class Store : public Unsafe::UnsafeStore<typename Dtype::Type, Shape::volume * threads, Device> {
			using dtype = typename Dtype::Type;
			
		public:

			Store() : Unsafe::UnsafeStore<dtype, Shape::volume* threads, Device>() {}
			Store(References::MutableShapedDeviceReference<dtype,Shape,Device> ptr) : Unsafe::UnsafeStore<dtype, Shape::volume* threads, Device>(ptr) {} //from a location

			References::MutableShapedDeviceReference<dtype,Shape, Device> asMutable(const Threads::ThreadContext& ctx) {
				return References::MutableShapedDeviceReference<dtype,Shape,Device>(Unsafe::UnsafeStore<dtype,Shape::volume*threads,Device>::asMutable() + (ctx.threadId * Shape::volume)); //pass to unsafe superclass
			}
			References::ImmutableShapedDeviceReference<dtype, Shape, Device> asImmutable(const Threads::ThreadContext& ctx) const {
				return References::ImmutableShapedDeviceReference<dtype,Shape,Device>(Unsafe::UnsafeStore<dtype, Shape::volume* threads, Device>::asImmutable() + (ctx.threadId * Shape::volume)); //pass to unsafe superclass
			}

			//copy a thread's data to another store
			template<Devices::Device target>
			void copyThreadTo(Store<Dtype,Shape,threads,target>& other, const Threads::ThreadContext& ctx) const {
				this->copyTo(other, ctx.threadId * Shape::volume * sizeof(dtype), Shape::volume * sizeof(dtype), ctx.stream);
			}

			//copy all data to another store
			template<Devices::Device target>
			inline void copyAllTo(Store<Dtype, Shape, threads, target>& other, const Threads::ThreadContext& ctx) const {
				this->template copyTo<target>(other, 0, Shape::volume * threads * sizeof(dtype), ctx.stream);
			}

		};
	}
}