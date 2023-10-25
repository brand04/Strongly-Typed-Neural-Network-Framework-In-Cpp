#pragma once
#include "../../dtypes/concept.h"
#include <concepts>
namespace NN {
	namespace MapFunctions {

		//determines if a Function is a function that can be used to map the Datatype to other values
		template<typename Function, typename Dtype>
		concept Mappable = 	std::is_void_v<decltype(Function::apply(std::declval<typename Dtype::Type&>()))>;
		
	}
}