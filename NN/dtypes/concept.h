#pragma once
#include <concepts>
namespace NN {
	namespace Dtypes {

		//concept of a neural network datatype - wrapper classes providing constuctors ect
		//outlines most of the constraits of a mathematical ring
		template<typename T>
		concept Dtype = requires {
			//check fields
			typename T::Type; //wraps a type of this value
			T::additiveIdentity;
			T::multiplicativeIdentity;

			//type check
			std::is_same<typename T::Type, decltype(T::additiveIdentity)>::value; //check that additiveIdentity exists and has the correct type
			std::is_same<typename T::Type, decltype(T::multiplicativeIdentity)>::value; //check that multiplicativeIdentity exists and has the correct type

			//check there exists a randomization or simply initialization function
			{ T::init(1) } -> std::same_as<typename T::Type>; //exists a method to randomize and that returns the correct type

	

			//check that the Dtype can be cast back into the wrapped type
			//{ static_cast<typename T::Type>(T()) } -> std::same_as<typename T::Type>; //exists an implicit conversion operator to T::Type

			

		};
	}
}