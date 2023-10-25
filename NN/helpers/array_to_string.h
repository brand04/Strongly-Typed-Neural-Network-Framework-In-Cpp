#pragma once
#include "../shapes/shape_concept.h"
#include "../references/immutable_reference.h"
#include <string>
namespace NN {
	namespace Helpers {
		//various functions for converting a shaped array to a string

		template<typename T, Shapes::IsShape Shape>
		static void arrayToString(std::stringstream& s, std::string prepend, References::ImmutableReference<T> arr) {
			s <<  "[";
			for (int i = 0; i < Shape::volume; i++) {
				s << arr[i];
				if (i != Shape::volume - 1 && i % Shape::asArray[Shape::dimension - 1] == Shape::asArray[Shape::dimension - 1] - 1) { //newline on lowest dimension
					s << "\n" << prepend;
				}
				else if (i != Shape::volume - 1) s << ", ";
			}
			s << "]";

		}

		template<typename T, Shapes::IsShape Shape>
		static void arrayToString(std::stringstream& s, std::string prepend, T* arr) {
			arrayToString<T,Shape>(s, prepend, References::ImmutableReference<T>(arr));
		}

		template<typename T, Shapes::IsShape Shape>
		static std::string arrayToString(T* arr) {
			std::stringstream s;
			arrayToString<T,Shape>(s, "", arr);
			return s.str();
		}

		template<typename T, Shapes::IsShape Shape>
		static std::string arrayToString(References::ImmutableReference<T> arr) {
			std::stringstream s;
			arrayToString<T,Shape>(s, "", arr);
			return s.str();
		}
	}
}