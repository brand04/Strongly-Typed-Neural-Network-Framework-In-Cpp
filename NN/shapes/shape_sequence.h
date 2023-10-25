#pragma once


namespace NN {

	namespace Shapes {

		template<typename ... Shapes>
		struct ShapeSequence {
		
		};


		namespace Operations {
			template<typename Seq, size_t i, typename Src>
			struct addToSequence;

			template<typename ... Shapes, size_t i , typename SrcShape>
			struct addToSequence<ShapeSequence<Shapes...>, i, SrcShape> {
				using type = addToSequence <ShapeSequence<Shapes..., SrcShape>,  i - 1, SrcShape>::type;
			};

			template<typename ... Shapes, typename SrcShape>
			struct addToSequence<ShapeSequence<Shapes...>, 0, SrcShape> {
				using type = ShapeSequence<Shapes...>;
			};
		}
		template<typename Shape, size_t n>
		using duplicateShape = typename Operations::addToSequence<ShapeSequence<>, n, Shape>::type;
	}
}