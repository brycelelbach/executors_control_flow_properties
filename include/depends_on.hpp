#pragma once

#include <tuple>

namespace detail
{
namespace depends_on_impl
{

template<class Derived>
struct property_base
{
  static constexpr bool is_requirable = true;
  static constexpr bool is_preferable = true;

  template<class Executor, class Type = decltype(Executor::query(*static_cast<Derived*>(0)))>
    static constexpr Type static_query_v = Executor::query(Derived());
};

} // end depends_on_impl
} // namespace detail

template<class... Executors>
class depends_on_t : public detail::depends_on_impl::property_base<depends_on_t<Executors...>>
{
  public:
    constexpr explicit depends_on_t(const Executors&... dependencies) : dependencies_(dependencies...) {}
    constexpr std::tuple<Executors...> value() const { return dependencies_; }

  private:
    std::tuple<Executors...> dependencies_;
};

template<>
class depends_on_t<void> : public detail::depends_on_impl::property_base<depends_on_t<void>>
{
  public:
    template<class... Executors>
    depends_on_t<Executors...> operator()(const Executors&... dependencies) const
    {
      return depends_on_t<Executors...>(dependencies...);
    }
};

constexpr depends_on_t<void> depends_on{};

