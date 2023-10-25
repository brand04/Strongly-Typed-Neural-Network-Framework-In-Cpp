#pragma once
#include <string>

namespace NN {
	/// <summary>
	/// Displays the network architecture
	/// </summary>
	/// <typeparam name="LayerT">The Type of the network</typeparam>
	template<typename LayerT>
	static constexpr std::string_view display = LayerT::template fullname<LayerT>;
}