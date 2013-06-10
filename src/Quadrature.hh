/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Pleas refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _QUADRATURE_HH_
#define _QUADRATURE_HH_

#include <cmath>
#include <vector>
#include "boost/math/special_functions/spherical_harmonic.hpp"
#include "Teuchos_BLAS.hpp"
#include "Teuchos_BLAS_types.hpp"
#include "Teuchos_LAPACK.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "deal.II/base/exceptions.h"


using namespace dealii;

typedef std::vector<double> d_vector;

/**
 * This class is the base class for the quadrature used for radiative
 * transfer.
 */

class Quadrature
{
  public :
    Quadrature(unsigned int sn,unsigned int L_max,bool galerkin);

    /// Build the quadrature, i.e. M,D and omega (direction vector).
    void build_quadrature(const double weight_sum);

    /// Return the number of directions of the quadrature.
    unsigned int get_n_dir() const;

    /// Return the number of moments.
    unsigned int get_n_mom() const;

    /// Return the degree l of expension coefficient of the scattering
    /// cross section given the number i of a moment.
    unsigned int get_l(const unsigned int i) const;

    /// Return a pointer to omega for direction idir.
    Teuchos::SerialDenseVector<int,double> const* const get_omega(unsigned int idir) 
      const;

    /// Return the moment-to-discrete matrix.
    Teuchos::SerialDenseMatrix<int,double> const* const get_M2D() const;

    /// Return the discrete-to-moment matrix.
    Teuchos::SerialDenseMatrix<int,double> const* const get_D2M() const;

  protected :
    /// Purely virtual function. Compute omega in one octant.
    virtual void build_octant() = 0;

    /// Deploy the octant.
    void deploy_octant();

    /// Compute the spherical harmonics and build the matrix M2D.
    void compute_harmonics(const double weight_sum);

    /// If flag is true, the quadrature is a Galerkin quadrature.
    const bool galerkin;
    /// Sn order of the quadrature.
    const unsigned int sn;
    /// L_max of the quadrature.
    const unsigned int L_max;
    /// Number of directions.
    unsigned int n_dir;
    /// Number of moments.
    unsigned int n_mom;
    /// Store the degree of the scattering cross section expansion associated
    /// to the number of the moment.
    d_vector moment_to_order;
    /// Weights of the quadrature when a non-Galerkin quadrature is used.
    Teuchos::SerialDenseVector<int,double> weight;
    /// Moments to directions matrix.
    Teuchos::SerialDenseMatrix<int,double> M2D;
    /// Directions to moments matrix.
    Teuchos::SerialDenseMatrix<int,double> D2M;
    /// Vector of omega for each direction.
    std::vector<Teuchos::SerialDenseVector<int,double> > omega;
};

inline unsigned int Quadrature::get_n_dir() const
{
  return n_dir;
}

inline unsigned int Quadrature::get_n_mom() const
{
  return n_mom;
}

inline unsigned int Quadrature::get_l(const unsigned int i) const
{
  return moment_to_order[i];
}

inline Teuchos::SerialDenseVector<int,double> const* const Quadrature::get_omega(unsigned int idir) const
{
  return &omega[idir];
}

inline Teuchos::SerialDenseMatrix<int,double> const* const Quadrature::get_M2D() const
{
  return &M2D;
}

inline Teuchos::SerialDenseMatrix<int,double> const* const Quadrature::get_D2M() const
{
  return &D2M;
}

#endif
