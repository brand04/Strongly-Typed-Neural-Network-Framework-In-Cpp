#pragma once
namespace NN{
	namespace Shapes{
		template<typename ShapeT>
		concept IsShape = requires {
			ShapeT::volume;
			ShapeT::dimension;
		};

	}
}