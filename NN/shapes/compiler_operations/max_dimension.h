#pragma once


namespace NN {
	namespace Shapes {
		namespace Operations {


			template<unsigned int value0, unsigned int value1, unsigned int ... values>
			struct maxValue;

			template<unsigned int value0, unsigned int value1>
			struct maxValue<value0, value1> {
				static constexpr unsigned int value = value0 > value1 ? value0 : value1;
			};

			template<unsigned int value0, unsigned int value1, unsigned int ... values>
			struct maxValue {
				static constexpr unsigned int cur_value = (value0 > value1 ? value0 : value1);
				static constexpr unsigned int next_value = maxValue<value1, values...>::value;
				static constexpr unsigned int value =  cur_value > next_value ? cur_value : next_value ;
			};

			

			template<typename Shape0, typename Shape1, typename ... Shapes>
			struct maxDimension {
				static constexpr unsigned int value = maxValue<Shape0::dimension, Shape1::dimension, Shapes::dimension...>::value;
			};

			
		}

		template<typename ... Shapes>
		static constexpr unsigned int getMaxDimension = Operations::maxDimension<Shapes...>::value;
	}
}