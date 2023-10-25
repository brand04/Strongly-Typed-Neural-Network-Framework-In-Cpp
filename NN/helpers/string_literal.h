#pragma once
#include <string>
namespace NN {

	/// <summary>
	/// A type representing a reference to a string literal stored elsewhere
	/// </summary>
	/// <typeparam name="length">The length of the referenced char[]</typeparam>
	template<size_t length>
	struct StringLiteral {

		template<typename T>
		using ref = T&;

		const ref<const char[length]> arr;
		static constexpr const size_t size = length;

		consteval StringLiteral(ref<const char[length]> str) : arr(str) {}

		consteval operator const std::string_view() const {
			return std::string_view(&(arr[0]), length);
		}

		consteval const std::string_view asStringView() const {
			return std::string_view(&(arr[0]), length);
		}

		operator ref<const char[length]>() {
			return arr;
		}
	};
}