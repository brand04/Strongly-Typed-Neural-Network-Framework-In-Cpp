#pragma once
#include <bit>

namespace NN {
	namespace Helpers {

		/// <summary>
		/// Flips the endian of value
		/// </summary>
		/// <typeparam name="T">type of value</typeparam>
		/// <param name="value">value to flip</param>
		/// <returns>value, with the endian flipped</returns>
		template<typename T>
		static constexpr T flipEndian(T& value) {
			T tmp = 0;
			for (int i = 0; i < sizeof(T); i++) {
				tmp |= ((value >> (8 * i)) & ((1 << 8) - 1)) << (8 * (sizeof(T) - i - 1));
			}
			return tmp;
		}

		/// <summary>
		/// Converts an input to the native endian, if not already
		/// </summary>
		/// <typeparam name="T">type of value</typeparam>
		/// <param name="val">value to make native endian</param>
		/// <param name="inputEndian">the endian of the value</param>
		/// <returns>val as native endian</returns>
		template<typename T>
		static constexpr inline T fromEndian(T val, const std::endian inputEndian) {
			if (inputEndian != std::endian::native) { //need to flip
				return flipEndian(val);
			}
			else return val;
		}

		/// <summary>
		/// Converts an input of native endian to an arbiary endian, if not already
		/// </summary>
		/// <typeparam name="T">type of the value</typeparam>
		/// <param name="val">value to make native endian</param>
		/// <param name="outputEndian">the endian to output</param>
		/// <returns>val as outputEndian</returns>
		template<typename T>
		static constexpr T toEndian(T val, const std::endian outputEndian) {
			if (outputEndian != std::endian::native) {
				return flipEndian(val);
			}
			else return val;
		}
	}
}