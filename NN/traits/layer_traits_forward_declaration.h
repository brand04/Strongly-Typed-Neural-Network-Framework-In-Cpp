#pragma once
namespace NN {
	namespace Traits {
		template<typename I_Dtype, typename I_Shape, typename I_Device, typename O_Dtype, typename O_Shape, typename O_Device, const unsigned int threads>
		struct LayerTraits;
	}
}