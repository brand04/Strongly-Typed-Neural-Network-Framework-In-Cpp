#pragma once
#include "immutable_reference.h"
namespace NN {
	namespace References {

		/// <summary>
		/// Wrapper around a pointer to non-const data
		/// Implicitely casts to void*, T*
		/// </summary>
		/// <typeparam name="T">The underlying type pointed to</typeparam>
		template<typename T>
		class MutableReference {
		public:
			T* ptr; //constant bits, but the underlying values may change
			MutableReference(T* location) : ptr(location) {}

			inline operator T* () {
				return ptr;
			}


			inline operator void* () {
				return ptr;
			}

			inline operator T const *() const {
				return ptr;
			}


			inline operator void* const () const {
				return ptr;
			}

			template<typename IntegralT>
			inline MutableReference<T> operator+(IntegralT integral) const {
				return MutableReference<T>(ptr + integral);
			}
		};
	}
}