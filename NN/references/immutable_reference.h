#pragma once
#include "../devices/includes.h"
namespace NN {
	namespace References {
		template<typename T>
		class ImmutableReference;

		template<typename T>
		class ImmutableReference {
		public:
				T const* ptr;

			ImmutableReference(const T* location) : ptr(location) {}

			inline operator T const *const () const {
				return ptr;
			}

			inline operator const void* () const {
				return ptr;
			}

			//force const-ness upon addition
			template<typename IntegralT>
			inline ImmutableReference<T> operator+ (IntegralT integral) {
				return ImmutableReference<T>(ptr + integral);
			}

		
		};
	}
}