/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _LS_HH_
#define _LS_HH_

#include "RTQuadrature.hh"

/**
 * This class build the Level Symmetric quadrature.
 */

class LS : public RTQuadrature
{
  public :
    LS(unsigned int sn,unsigned int L_max,bool galerkin);

  private :
    /// Compute omega in one octant.
    void build_octant();

    /// Compute the different omega given a set of directions.
    void compute_omega(std::vector<double> const &direction);
};

#endif
