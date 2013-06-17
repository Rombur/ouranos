/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "TimeDiscretization.hh"

TimeDiscretization::TimeDiscretization(DISCRETIZATION_METHOD method,
    double time_step,double final_time) :
  time_step(time_step),
  final_time(final_time)
{
  switch (method)
  {
    case EXPLICIT_EULER :
      {
        b_tableau.a.resize(1,d_vector(1,0.));
        b_tableau.b.push_back(1.);
        b_tableau.c.push_back(0.);
        break;
      }
    case IMPLICIT_EULER :
      {
        b_tableau.a.resize(1,d_vector(1,1.));
        b_tableau.b.push_back(1.);
        b_tableau.c.push_back(0.);
        break;
      }
    case CRANK_NICHOLSON :
      {
        b_tableau.a.resize(2,d_vector(2,0.));
        b_tableau.b.resize(2,0.);
        b_tableau.c.resize(2,0.);
        b_tableau.a[1][0] = 0.5;
        b_tableau.a[1][1] = 0.5;
        b_tableau.b[0] = 0.5;
        b_tableau.b[1] = 0.5;
        b_tableau.c[1] = 1.;
        break;
      }
    case RK44 :
      {
        b_tableau.a.resize(4,d_vector(4,0.));
        b_tableau.b.resize(4,0.);
        b_tableau.c.resize(4,0.);
        b_tableau.a[1][0] = 0.5;
        b_tableau.a[2][1] = 0.5;
        b_tableau.a[3][2] = 1.;
        b_tableau.b[0] = 1./6.;
        b_tableau.b[1] = 1./3.;
        b_tableau.b[2] = 1./3.;
        b_tableau.b[3] = 1./6.;
        b_tableau.c[1] = 0.5;
        b_tableau.c[2] = 0.5;
        b_tableau.c[3] = 1.;
        break;      
      }
    case IRK42 :
      {
        const double sqrt_3(std::sqrt(3));
        b_tableau.a.resize(2,d_vector(2,0.));
        b_tableau.b.resize(2,0.);
        b_tableau.c.resize(2,0.);
        b_tableau.a[0][0] = 1./4.;
        b_tableau.a[0][1] = 1./4.-sqrt_3/6.;
        b_tableau.a[1][0] = 1./4.+sqrt_3/6.;
        b_tableau.a[1][1] = 1./4.;
        b_tableau.b[0] = 0.5;
        b_tableau.b[1] = 0.5;
        b_tableau.c[0] = 0.5-sqrt_3/6.;
        b_tableau.c[1] = 0.5+sqrt_3/6.;
        break;
      }
  }
}
