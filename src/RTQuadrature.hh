/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _RTQUADRATURE_HH_
#define _RTQUADRATURE_HH_

#include <unordered_set>
#include <vector>
#include "deal.II/base/exceptions.h"
#include "deal.II/lac/full_matrix.h"
#include "deal.II/lac/vector.h"


using namespace dealii;


/**
 * This class is the base class for the quadrature used for radiative
 * transfer.
 */

class RTQuadrature
{
  public :
    RTQuadrature(unsigned int sn,unsigned int L_max,bool galerkin);

    virtual ~RTQuadrature() {};

    /// Build the quadrature, i.e. M, D, and omega (direction vector).
    void build_quadrature(const double weight_sum,const unsigned int dim);

    /// Return true if the direction is one of the most normal to the
    /// boundaries.
    bool is_most_normal_direction(unsigned int face,unsigned int idir) const;

    /// Return the number of directions of the quadrature.
    unsigned int get_n_dir() const;

    /// Return the number of moments.
    unsigned int get_n_mom() const;

    /// Return the degree l of expension coefficient of the scattering
    /// cross section given the number i of a moment.
    unsigned int get_l(const unsigned int i) const;

    /// Return a pointer to omega for direction idir.
    Vector<double> const* const get_omega(unsigned int idir) const;

    /// Return the moment-to-discrete matrix.
    FullMatrix<double> const* const get_M2D() const;

    /// Return the discrete-to-moment matrix.
    FullMatrix<double> const* const get_D2M() const;

  protected :
    /// Purely virtual function. Compute omega in one octant.
    virtual void build_octant() = 0;

    /// Deploy the octant.
    void deploy_octant();

    /// Compute the spherical harmonics and build the matrix M2D.
    void compute_harmonics(const double weight_sum);

    /// Compute the most normal directions to the boundaries.
    void compute_most_normal_directions(const unsigned int dim);

    /// If flag is true, the quadrature is a Galerkin quadrature. Galerkin
    /// quadrature is only implemented for 2D triangular quadratures.
    const bool galerkin;
    /// Sn order of the quadrature.
    const unsigned int sn;
    /// L_max of the quadrature.
    const unsigned int L_max;
    /// Number of directions.
    unsigned int n_dir;
    /// Number of moments.
    unsigned int n_mom;
    /// Set of the most normal directions to the boundary.
    std::vector<std::unordered_set<unsigned int>> most_n_directions;
    /// Store the degree of the scattering cross section expansion associated
    /// to the number of the moment.
    std::vector<double> moment_to_order;
    /// Weights of the quadrature when a non-Galerkin quadrature is used.
    Vector<double> weight;
    /// Moments to directions matrix.
    FullMatrix<double> M2D;
    /// Directions to moments matrix.
    FullMatrix<double> D2M;
    /// Vector of omega for each direction.
    std::vector<Vector<double>> omega;
};

inline bool RTQuadrature::is_most_normal_direction(unsigned int face,unsigned int idir) const
{
  return ((most_n_directions[face].count(idir)==0) ? false : true);
}

inline unsigned int RTQuadrature::get_n_dir() const
{
  return n_dir;
}

inline unsigned int RTQuadrature::get_n_mom() const
{
  return n_mom;
}

inline unsigned int RTQuadrature::get_l(const unsigned int i) const
{
  return moment_to_order[i];
}

inline Vector<double> const* const RTQuadrature::get_omega(unsigned int idir) const
{
  return &omega[idir];
}

inline FullMatrix<double> const* const RTQuadrature::get_M2D() const
{
  return &M2D;
}

inline FullMatrix<double> const* const RTQuadrature::get_D2M() const
{
  return &D2M;
}

#endif
