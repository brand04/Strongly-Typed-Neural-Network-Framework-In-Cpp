#pragma once
#include <cuda_runtime.h>
#include "activation_macro.h"

namespace NN {
	namespace Activations {
		//Relu defined with macro 
		//Name, apply function, derivative function
		__ACTIVATION_FUNCTION__(ReluActivation, return (x > 0) ? x : 0; , return (x > 0) ? 1 : 0; )

	}
}