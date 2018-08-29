#include <new_thread_executor.hpp>
#include <lazy_oneway_executor_adaptor.hpp>
#include <chrono>
#include <iostream>
#include <cassert>

bool set_me = false;

int main()
{
  new_thread_executor ex;

  lazy_oneway_executor_adaptor<new_thread_executor> lazy_ex(ex);

  lazy_ex.execute([]
  {
    set_me = true;
  });

  // ensure that set_me remains false for half a second
  auto half_a_second_from_now = std::chrono::system_clock::now() + std::chrono::milliseconds(500);

  while(std::chrono::system_clock::now() < half_a_second_from_now)
  {
    assert(set_me == false);
  }

  // flush the work
  lazy_ex.flush();

  // wait half a second for the flag to be set
  half_a_second_from_now = std::chrono::system_clock::now() + std::chrono::milliseconds(500);
  while(std::chrono::system_clock::now() < half_a_second_from_now) {}

  assert(set_me);

  std::cout << "OK" << std::endl;

  return 0;
}

