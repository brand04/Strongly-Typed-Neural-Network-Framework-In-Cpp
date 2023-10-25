#pragma once
#include <tuple>

namespace NN {

	/// <summary>
	/// A type representing a fixed length owned string - ie the string is not a string literal and is constructed at compile time
	/// </summary>
	/// <typeparam name="length">length of the string</typeparam>
	template<size_t length>
	struct FixedString {
		char arr[length];
		static const constexpr size_t size = length;
		consteval const std::string_view asString() const {
			return std::string_view(arr, length);
		}

		consteval operator const std::string_view() const {
			return asString();
		}

		constexpr const std::string_view asStringView() const {
			return std::string_view(&(arr[0]), length);
		}
	};

}