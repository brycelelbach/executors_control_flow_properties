#include <lightweight_test.hpp>

#include <inline_executor.hpp>

int main()
{
  inline_executor ie;

  ie.execute([] { printf("1st\n"); });
  ie.execute([] { printf("2nd\n"); });
  ie.execute([] { printf("3rd\n"); });

  ie.wait(); // No-op; should be preferred not required.
}

