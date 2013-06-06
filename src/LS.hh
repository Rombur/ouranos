/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Pleas refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _LS_HH_
#define _LS_HH_

#include "Quadrature.hh"

/**
 * This class build the Level Symmetric quadrature.
 */

class LS : public Quadrature
{
  public :
    LS(unsigned int sn,unsigned int L_max,bool galerkin);

  private :
    /// Compute omega in one octant.
    void build_octant();

    /// Compute the different omega given a set of directions.
    void compute_omega(d_vector const &direction);
};

#endif
