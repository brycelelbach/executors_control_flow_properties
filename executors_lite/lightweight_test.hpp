// Based on boost/core/lightweight_test.hpp
// Imported from: git@github.com:boostorg/core.git, f1c51cb, retrieved 2016-10-25
//
// Copyright (c) 2018    NVIDIA Corporation
// Copyright (c) 2016    Bryce Adelstein Lelbach aka wash
// Copyright (c) 2002-14 Peter Dimov
// Copyright (c) 2010-11 Beman Dawes
// Copyright (3) 2013    Ion Gaztanaga
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#ifndef EXECUTORS_LITE_CORE_LIGHTWEIGHT_TEST_HPP
#define EXECUTORS_LITE_CORE_LIGHTWEIGHT_TEST_HPP

#include <iostream>
#include <cassert>

#include <executors_lite/preprocessor.hpp>

//  IDE's like Visual Studio perform better if output goes to std::cout or
//  some other stream, so allow user to configure output stream:
#ifndef EL_LIGHTWEIGHT_TEST_OSTREAM
# define EL_LIGHTWEIGHT_TEST_OSTREAM std::cerr
#endif

namespace el
{

namespace detail
{

struct report_errors_reminder
{
    bool called_report_errors_function;

    report_errors_reminder() : called_report_errors_function(false) {}

    ~report_errors_reminder()
    {
        assert(called_report_errors_function);  // verify report_errors() was called  
    }
};

inline report_errors_reminder& report_errors_remind()
{
    static report_errors_reminder r;
    return r;
}

inline int & test_errors()
{
    static int x = 0;
    report_errors_remind();
    return x;
}

inline void test_failed_impl(char const * expr, char const * file, int line, char const * function)
{
    EL_LIGHTWEIGHT_TEST_OSTREAM
      << file << "(" << line << "): test '" << expr << "' failed in function '"
      << function << "'" << std::endl;
    ++test_errors();
}

inline void error_impl(char const * msg, char const * file, int line, char const * function)
{
    EL_LIGHTWEIGHT_TEST_OSTREAM
      << file << "(" << line << "): " << msg << " in function '"
      << function << "'" << std::endl;
    ++test_errors();
}

inline void throw_failed_impl(char const * excep, char const * file, int line, char const * function)
{
   EL_LIGHTWEIGHT_TEST_OSTREAM
    << file << "(" << line << "): Exception '" << excep << "' not thrown in function '"
    << function << "'" << std::endl;
   ++test_errors();
}

template<class T, class U> inline void test_eq_impl( char const * expr1, char const * expr2,
  char const * file, int line, char const * function, T const & t, U const & u )
{
    if( t == u )
    {
        report_errors_remind();
    }
    else
    {
        EL_LIGHTWEIGHT_TEST_OSTREAM
            << file << "(" << line << "): test '" << expr1 << " == " << expr2
            << "' failed in function '" << function << "': "
            << "'" << t << "' != '" << u << "'" << std::endl;
        ++test_errors();
    }
}

template<class T, class U> inline void test_ne_impl( char const * expr1, char const * expr2,
  char const * file, int line, char const * function, T const & t, U const & u )
{
    if( t != u )
    {
        report_errors_remind();
    }
    else
    {
        EL_LIGHTWEIGHT_TEST_OSTREAM
            << file << "(" << line << "): test '" << expr1 << " != " << expr2
            << "' failed in function '" << function << "': "
            << "'" << t << "' == '" << u << "'" << std::endl;
        ++test_errors();
    }
}

} // namespace detail

inline int report_errors()
{
    el::detail::report_errors_remind().called_report_errors_function = true;

    int errors = el::detail::test_errors();

    if( errors == 0 )
    {
        EL_LIGHTWEIGHT_TEST_OSTREAM
          << "No errors detected." << std::endl;
        return 0;
    }
    else
    {
        EL_LIGHTWEIGHT_TEST_OSTREAM
          << errors << " error" << (errors == 1? "": "s") << " detected." << std::endl;
        return 1;
    }
}

} // namespace el

#define EL_TEST(expr) ((expr)? (void)0: ::el::detail::test_failed_impl(#expr, __FILE__, __LINE__, CURRENT_FUNCTION))
#define EL_TEST_NOT(expr) EL_TEST(!(expr))

#define EL_ERROR(msg) ( ::el::detail::error_impl(msg, __FILE__, __LINE__, CURRENT_FUNCTION) )

#define EL_TEST_EQ(expr1,expr2) ( ::el::detail::test_eq_impl(#expr1, #expr2, __FILE__, __LINE__, CURRENT_FUNCTION, expr1, expr2) )
#define EL_TEST_NE(expr1,expr2) ( ::el::detail::test_ne_impl(#expr1, #expr2, __FILE__, __LINE__, CURRENT_FUNCTION, expr1, expr2) )

#endif // #ifndef EXECUTORS_LITE_CORE_LIGHTWEIGHT_TEST_HPP

