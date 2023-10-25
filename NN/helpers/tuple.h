#pragma once
#include "../asserts/false_assert.h"
#include <tuple>
namespace NN {
	namespace Helpers {

		//std::tuple caused problems, i believe to do with move/copy semantics but i'm not yet smart enough to know why, just that at the end of a constructor call pointer values were changed seemingly randomly
		//this has in-place construction so that it can be constructed without copy/move

		//used in sequences/concurrent/concurrent_sequence.h
		

		template<typename ... Ts>
		struct ArgPack {
			std::tuple<Ts...> args;
		};

		
		template<typename ...>
		struct Tuple;

		template<typename T0, typename ... Ts>
		struct Tuple<T0,Ts...> {
			T0 head;
			Tuple<Ts...> tail;
			template<typename ... Arg0s, typename ... ArgPacks>
			Tuple(const ArgPack<Arg0s...>&& pack0, const ArgPacks&&... packs) : Tuple(std::make_index_sequence<sizeof...(Arg0s)>(),std::move(pack0),std::move(packs)...) {}
			template<size_t ... indicies, typename ... Arg0s, typename ... ArgPacks>
			Tuple(const std::index_sequence<indicies...>, const ArgPack<Arg0s...>&& pack0, const ArgPacks&& ... packs) : head(std::move(std::get<indicies>(std::move(pack0.args)))...), tail(std::move(packs)...) {}

			Tuple() : head(), tail() {}
		};

		template<typename T0>
		struct Tuple<T0> {
			T0 head;
			template<typename ... Arg0s>
			Tuple(const ArgPack<Arg0s...>&& pack0) : Tuple(std::make_index_sequence<sizeof...(Arg0s)>(), std::move(pack0)) {}
			template<size_t... indicies, typename ... Arg0s>
			Tuple(std::index_sequence<indicies...>, const ArgPack<Arg0s...>&& pack0) : head(std::move(std::get<indicies>(std::move(pack0.args)))...) {}

			Tuple() : head() {}
		};

		
		template<typename Tup, size_t n>
		struct getElement;

		template<typename T0, typename ... Ts, size_t n>
		struct getElement<Tuple<T0,Ts...>, n> {
			using type = getElement<Tuple<Ts...>, n - 1>::type;
			static constexpr inline type& getElementValue(Tuple<T0,Ts...>& tuple) {
				return getElement<Tuple<Ts...>, n - 1>::getElementValue(tuple.tail);
			}
			static constexpr inline const type& getElementValue(const Tuple<T0,Ts...>& tuple) {
				return getElement<Tuple<Ts...>, n - 1>::getElementValue(tuple.tail);
			}
		};

		template<typename T0, typename ... Ts>
		struct getElement<Tuple<T0, Ts...>, 0> {
			using type = T0;
			static constexpr inline type& getElementValue(Tuple<T0, Ts...>& tuple) {
				return tuple.head;
			}
			static constexpr inline const type& getElementValue(const Tuple<T0, Ts...>& tuple) {
				return tuple.head;
			}
		};

		template<size_t n>
		struct getElement<Tuple<>, n> {
			static_assert(struct_assert < Asserts::AssertFalseWith<std::index_sequence<n>>>, "Index out of bounds");
		};

		template<size_t n, typename TupleT>
		static constexpr inline auto& get(const TupleT& tuple) {
			return getElement<TupleT,n>::getElementValue(tuple);
		}

		template<size_t n, typename TupleT>
		static constexpr inline auto& get(TupleT& tuple) {
			return getElement<TupleT,n>::getElementValue(tuple);
		}

		template<size_t n, typename TupleT>
		using getType = getElement<TupleT, n>::type;

	}
}