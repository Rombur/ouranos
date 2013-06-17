/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _TIMEDISCRETIZATION_HH_
#define _TIMEDISCRETIZATION_HH_

#include <cmath>
#include <vector>
#include "deal.II/base/exceptions.h"

using namespace dealii;
typedef std::vector<double> d_vector;

/// Enum on the following time discretization methods: explicit Euler, 
/// implicit Euler, Crank-Nicholson, explicit Runge-Kutta of order 4 and 
/// 4 stages, implicit Runge-Kutta of order 4 and 2 stages
enum DISCRETIZATION_METHOD{EXPLICIT_EULER,IMPLICIT_EULER,CRANK_NICHOLSON,RK44,
  IRK42};

/// Structure that contains the Butcher tableau of the discretization method.
struct ButcherTableau
{
  public :
    d_vector b;
    d_vector c;
    std::vector<d_vector> a;
};

/**
 * This class builds the Butcher tableau of the time discretization methods.
 * Only methods that can be reprensented by a Butcher tableau can be used.
 */

class TimeDiscretization
{
  public :
    TimeDiscretization(DISCRETIZATION_METHOD method,double time_step,
        double final_time);

    /// Return true if the method is implicit.
    bool is_implicit() const;

    /// Return the current time step.
    double get_time_step() const;

    /// Return the end time of the simulation.
    double get_final_time() const;

    /// Return the Butcher coefficient a(i,j).
    double get_butcher_a(unsigned int i,unsigned int j) const;

    /// Return the Butcher coefficient b(i).
    double get_butcher_b(unsigned int i) const;

    /// Return the Butcher coefficient c(i).
    double get_butcher_c(unsigned int j) const;

  private :
    /// Flag to indicate if the method is implicit.
    bool implicit;
    /// Time step.
    double time_step;
    /// End time of the simulation.
    double final_time;
    /// Butcher tableau.
    ButcherTableau b_tableau;
};

inline bool is_implicit() const
{
  return implicit;
}

inline double TimeDiscretization::get_time_step() const
{
  return time_step;
}

inline double TimeDiscretization::get_final_time() const
{
  return final_time;
}

inline double TimeDiscretization::get_butcher_a(unsigned int i,unsigned int j) const
{
  AssertIndexRange(i,b_tableau.a.size());
  AssertIndexRange(j,b_tableau.a[i].size());
  return b_tableau.a[i][j];
}

inline double TimeDiscretization::get_butcher_b(unsigned int i) const
{
  AssertIndexRange(i,b_tableau.b.size());
  return b_tableau.b[i];
}

inline double TimeDiscretization::get_butcher_c(unsigned int i) const
{
  AssertIndexRange(i,b_tableau.c.size());
  return b_tableau.c[i];
}

#endif
