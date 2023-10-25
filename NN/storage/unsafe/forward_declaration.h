#pragma once
#include "../../devices/includes.h"
namespace NN {
	namespace Storage {
		namespace Unsafe {

			/// <summary>
			/// Allocates storage for 'size' elements of type 'T' on the specified device
			/// </summary>
			/// <typeparam name="T">type of the elements</typeparam>
			/// <typeparam name="size">number of elements</typeparam>
			/// <typeparam name="device">Device to store on</typeparam>
			template<typename T, const size_t size, Devices::Device device>
			class UnsafeStore;
		}
	}
}