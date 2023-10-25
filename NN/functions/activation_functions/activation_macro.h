#pragma once
#include <cuda_runtime.h>
#include "../../helpers/string_literal.h"
namespace NN {
	namespace Activations {
		
		#define __ACTIVATION_FUNCTION_DEVICES__ __host__ __device__
		#define __ACTIVATION_FUNCTION__(NAME,APPLY,DERIVATIVE) struct NAME final { \
			static constexpr StringLiteral name = StringLiteral(#NAME); \
			template<typename T> \
			__ACTIVATION_FUNCTION_DEVICES__ static T apply(const T x) const { APPLY \
			} \
			template<typename T> \
			__ACTIVATION_FUNCTION_DEVICES__ static T derivative(const T x) { DERIVATIVE } \
		}; \

	}
}