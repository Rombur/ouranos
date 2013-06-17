/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <cmath>
#include "../src/TimeDiscretization.hh"

// Solve the problem dy/dt=ty, y(0) = 1, solution = exp(t^2/2)
TEST_CASE("TimeDiscretization/Explicit Euler","Check explicit Euler")
{
  const double delta_t(0.001);
  const double final_time(1.);
  TimeDiscretization time(EXPLICIT_EULER,delta_t,final_time);

  REQUIRE(time.get_time_step()==delta_t);
  const unsigned int n_step(final_time/delta_t);
  double y(1.);
  for (unsigned int i=1; i<n_step; ++i)
  {
    double k0(delta_t*y*i*delta_t);
    y += time.get_butcher_b(0)*k0;
    REQUIRE(std::fabs(y-std::exp(std::pow(i*delta_t,2) /2.))<1e-3);
  }
}
