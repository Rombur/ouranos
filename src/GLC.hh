/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Pleas refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _GLC_HH_
#define _GLC_HH_

#include <cmath>
#include <vector>
#include "Quadrature.hh"


typedef std::vector<double> d_vector;

/**
 * This class build the triangular Gauss-Legendre-Chebyshev quadrature, i.e.
 * \f$\cos(\theta)\f$ (\f$\theta is the polar angle\f$) uses a Gauss-Legendre
 * quadrature while the azimuthal angle \f$\phi\f$ uses a Chebyshev
 * quadrature.
 */ 

class GLC : public Quadrature
{
  public : 
    GLC(unsigned int sn,unsigned int L_max,bool galerkin);

  private :
    /// Compute omega in one octant.
    void build_octant();

    /// Build the Chebyshev quadrature.
    void build_chebyshev_quadrature(d_vector &nodes,d_vector &weight);

    /// Build the Gauss-Legendre quadrature.
    void build_gauss_legendre_quadrature(d_vector &nodes,d_vector &weight);
};

#endif
