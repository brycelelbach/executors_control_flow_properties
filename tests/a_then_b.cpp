// case 1: ExecutorB knows how to depend on ExecutorA
template<class ExecutorA, class FunctionA, class ExecutorB, class FunctionB,
         __REQUIRES(can_require_v<ExecutorB, depends_on_t<ExecutorA, single_t>)
        >
auto a_then_b(const ExecutorA& ex_a, FunctionA&& f_a, const ExecutorB& ex_b, FunctionB&& f_b)
{
  using namespace std::experimental::execution;

  ex_a.execute(std::forward<FunctionA>(f_a));

  return require(ex_b, depends_on(ex_a)).execute(std::forward<FunctionB>(f_b));
}

// case 2: not case 1 and ExecutorB can be lazy and forgetful
template<class ExecutorA, class FunctionA, class ExecutorB, class FunctionB,
         __REQUIRES(!can_require_v<ExecutorB, depends_on_t<ExecutorA, single_t>),
         __REQUIRES(can_require_v<ExecutorB, lazy_t, depends_on_t<>, single_t>)
        >
auto a_then_b(const ExecutorA& ex_a, FunctionA&& f_a, const ExecutorB& ex_b, FunctionB&& f_b)
{
  using namespace std::experimental::execution;

  // forget ex_b's dependencies and make it a lazy then executor
  auto lazy_ex_b = require(ex_b, depends_on(), lazy, then);

  // lazily enqueue f_b
  auto result_ex = lazy_ex_b.execute(f_b);

  // execute f_a on ex_a and flush lazy_ex_b after f_a() finishes
  ex_a.execute([f_a = std::forward<FunctionA>(f_a), lazy_ex_b = std::move(lazy_ex_b)]
  {
    // invoke f_a
    f_a();

    // release lazy_ex_b's work (which is f_b)
    lazy_ex_b.flush();
  });

  return result_ex;
}

