#pragma once
#include "./fixed_string.h"
#include "./string_literal.h"
#include "../asserts/false_assert.h"
#include <string>
#include <tuple>
namespace NN {

	//forward definition
	template<typename ... StrTypes>
	struct StringCollection;


	namespace StringCollectionFunctions {
		//forward declaration that fails if no specialization matches
		template<typename StrTypePtr, size_t size>
		constexpr static void copyTo(StrTypePtr ptr, char arr[size], const size_t offset) {
			static_assert(struct_assert<Asserts::AssertFalseWith<StrTypePtr>>, "Failed to copy - type not supported");
		}

		template<typename T>
		using ref = T&;



		//functions for copying from various string types to a buffer at compile time


		/// <summary>
		/// Copies the contents of slice into the contents of arr at the specified offset, AND increments the offset by the amount copied
		/// </summary>
		/// <typeparam name="sliceSize">The size of the slice to copy</typeparam>
		/// <typeparam name="size">The size of the destination array</typeparam>
		/// <param name="slice">The slice</param>
		/// <param name="arr">The destination array</param>
		/// <param name="offset">The offset</param>
		template<size_t sliceSize, size_t size>
		constexpr static void copyTo(const ref<const char[sliceSize]> slice, ref<char[size]> arr, size_t& offset) {
			for (size_t i = 0; i < sliceSize-1; i++) {
				arr[i + offset] = slice[i];
			}
			offset += sliceSize - 1;
		}
		template<size_t sliceSize, size_t size>
		constexpr static void copyTo(const StringLiteral<sliceSize>& str, ref<char[size]> arr, size_t& offset) {
			copyTo(str.arr, arr, offset);
		}

		template<size_t sliceSize, size_t size>
		constexpr static void copyTo(const FixedString<sliceSize>& str, ref<char[size]> arr, size_t& offset) {
			copyTo(str.arr, arr, offset);
		}

		template<typename ... types, size_t size>
		constexpr static void copyTo(StringCollection<types...> node, ref<char[size]> arr, size_t& offset) { 
			node.copyToWithOffset(arr, offset, node.ptrs);
		}

		template<typename> struct LengthOf;

		//pretty sure theres a way to not have to specialize for constness and non-constness

		//Get the length of a string type

		template<size_t size>
		struct LengthOf<StringLiteral<size>> {
			static constexpr size_t value = size;
		};
		template<size_t size>
		struct LengthOf<const StringLiteral<size>> {
			static constexpr size_t value = size;
		};
		template<size_t size>
		struct LengthOf<FixedString<size>> {
			static constexpr size_t value = size;
		};
		template<size_t size>
		struct LengthOf<const FixedString<size>> {
			static constexpr size_t value = size;
		};
		template<size_t size>
		struct LengthOf<const char[size]> {
			static constexpr size_t value = size;
		};

		template<size_t size>
		struct LengthOf<char[size]> {
			static constexpr size_t value = size;
		};

		//get the length of a StringCollection of types
		template<typename ... StrTypes>
		struct LengthOf<StringCollection<StrTypes...>> {
			static constexpr size_t value = StringCollection<StrTypes...>::length;
		};

		template<typename ... StrTypes>
		struct LengthOf<const StringCollection<StrTypes...>> {
			static constexpr size_t value = StringCollection<StrTypes...>::length;
		};

		template<typename ErrorType>
		struct LengthOf {
			static_assert(struct_assert<Asserts::AssertFalseWith<ErrorType>>, "Invalid type supplied");
		};


	}

	/// <summary>
	/// A Tree-like structure linking multiple types and instances of different String Types (const (char&)[], StringLiteral<>, FixedString<>, StringCollection<...>)
	/// </summary>
	/// <typeparam name="...StrTypes">The Types contained</typeparam>
	template<typename ... StrTypes>
	struct StringCollection {
	private:
		template<typename T>
		using ref = T&;
		
		

		template<size_t ... indices, size_t size>
		constexpr void copyToImpl(ref<char[size]> arr, std::index_sequence<indices...> seq, size_t& offset, const std::tuple<const StrTypes&...> ptrs) const {

			(StringCollectionFunctions::copyTo(std::get<indices>(ptrs), arr, offset),...);
		}
	public:

		const std::tuple<const StrTypes&...> ptrs;

		
		constexpr const StringCollection(const StrTypes&... strs) : ptrs(strs...) {}

		//store the length at each node in the tree for easier recursion
		static constexpr size_t length = (0 + ... + StringCollectionFunctions::LengthOf<StrTypes>::value) - sizeof...(StrTypes) + 1; //adjust for null bytes


		template<size_t size>
		constexpr void copyToWithOffset(ref<char[size]> arr, size_t& offset, const std::tuple<const StrTypes&...> ptrs) const {
			copyToImpl(arr, std::make_index_sequence<sizeof...(StrTypes)>(), offset, ptrs);
		}

		template<size_t size>
		constexpr void copyTo(ref<char[size]> arr, const std::tuple<const StrTypes&...> ptrs) const {
			size_t offset = 0;
			copyToWithOffset(arr, offset, ptrs);
		}

		consteval const FixedString<length> fix() const {
			FixedString<length> result;
			copyTo(result.arr, ptrs);
			result.arr[length - 1] = '\0'; //null byte
			return result;
		}
	};
	

}