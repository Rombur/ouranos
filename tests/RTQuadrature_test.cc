/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Pleas refer to the file
 * license.txt for the text and further information on this license.
 */

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iomanip>
#include <cmath>
#include <vector>
#include "deal.II/lac/full_matrix.h"
#include "deal.II/lac/vector.h"
#include "../src/GLC.hh"
#include "../src/LS.hh"

// is_most_normal_direction

TEST_CASE("RTQuadrature/LS","Check LS quadrature")
{
  unsigned int n_dir(12);
  const double four_pi(4.*M_PI);
  std::vector<double> omega(3,0.);
  omega[0] = 0.868890300722201205229788;
  omega[1] = 0.350021174581540677777041;
  omega[2] = 0.350021174581540677777041;

  LS quad(4,4,true);
  quad.build_quadrature(four_pi,2);

  // Check the most normal directions.
  REQUIRE(quad.is_most_normal_direction(0,0)==true);
  REQUIRE(quad.is_most_normal_direction(0,1)==false);
  REQUIRE(quad.is_most_normal_direction(0,2)==false);
  REQUIRE(quad.is_most_normal_direction(0,3)==true);
  REQUIRE(quad.is_most_normal_direction(0,4)==false);
  REQUIRE(quad.is_most_normal_direction(0,5)==false);
  REQUIRE(quad.is_most_normal_direction(0,6)==false);
  REQUIRE(quad.is_most_normal_direction(0,7)==false);
  REQUIRE(quad.is_most_normal_direction(0,8)==false);
  REQUIRE(quad.is_most_normal_direction(0,9)==false);
  REQUIRE(quad.is_most_normal_direction(0,10)==false);
  REQUIRE(quad.is_most_normal_direction(0,11)==false);
  REQUIRE(quad.is_most_normal_direction(1,0)==false);
  REQUIRE(quad.is_most_normal_direction(1,1)==false);
  REQUIRE(quad.is_most_normal_direction(1,2)==false);
  REQUIRE(quad.is_most_normal_direction(1,3)==false);
  REQUIRE(quad.is_most_normal_direction(1,4)==false);
  REQUIRE(quad.is_most_normal_direction(1,5)==false);
  REQUIRE(quad.is_most_normal_direction(1,6)==true);
  REQUIRE(quad.is_most_normal_direction(1,7)==false);
  REQUIRE(quad.is_most_normal_direction(1,8)==false);
  REQUIRE(quad.is_most_normal_direction(1,9)==true);
  REQUIRE(quad.is_most_normal_direction(1,10)==false);
  REQUIRE(quad.is_most_normal_direction(1,11)==false);
  REQUIRE(quad.is_most_normal_direction(2,0)==false);
  REQUIRE(quad.is_most_normal_direction(2,1)==true);
  REQUIRE(quad.is_most_normal_direction(2,2)==false);
  REQUIRE(quad.is_most_normal_direction(2,3)==false);
  REQUIRE(quad.is_most_normal_direction(2,4)==false);
  REQUIRE(quad.is_most_normal_direction(2,5)==false);
  REQUIRE(quad.is_most_normal_direction(2,6)==false);
  REQUIRE(quad.is_most_normal_direction(2,7)==false);
  REQUIRE(quad.is_most_normal_direction(2,8)==false);
  REQUIRE(quad.is_most_normal_direction(2,9)==false);
  REQUIRE(quad.is_most_normal_direction(2,10)==true);
  REQUIRE(quad.is_most_normal_direction(2,11)==false);
  REQUIRE(quad.is_most_normal_direction(3,0)==false);
  REQUIRE(quad.is_most_normal_direction(3,1)==false);
  REQUIRE(quad.is_most_normal_direction(3,2)==false);
  REQUIRE(quad.is_most_normal_direction(3,3)==false);
  REQUIRE(quad.is_most_normal_direction(3,4)==true);
  REQUIRE(quad.is_most_normal_direction(3,5)==false);
  REQUIRE(quad.is_most_normal_direction(3,6)==false);
  REQUIRE(quad.is_most_normal_direction(3,7)==true);
  REQUIRE(quad.is_most_normal_direction(3,8)==false);
  REQUIRE(quad.is_most_normal_direction(3,9)==false);
  REQUIRE(quad.is_most_normal_direction(3,10)==false);
  REQUIRE(quad.is_most_normal_direction(3,11)==false);

  // Check the number of direction
  REQUIRE(n_dir==quad.get_n_dir());

  // Check the number of moments
  REQUIRE(n_dir==quad.get_n_mom());

  // Check the degree of the expansion
  REQUIRE(quad.get_l(0)==0);
  REQUIRE(quad.get_l(1)==1);
  REQUIRE(quad.get_l(2)==1);
  REQUIRE(quad.get_l(3)==2);
  REQUIRE(quad.get_l(4)==2);
  REQUIRE(quad.get_l(5)==2);
  REQUIRE(quad.get_l(6)==3);
  REQUIRE(quad.get_l(7)==3);
  REQUIRE(quad.get_l(8)==3);
  REQUIRE(quad.get_l(9)==3);
  REQUIRE(quad.get_l(10)==4);
  REQUIRE(quad.get_l(11)==4);

  // Check omega and omega_2d
  Vector<double> const* const omega_ptr(quad.get_omega(0));
  REQUIRE(omega[0]==(*omega_ptr)(0));
  REQUIRE(omega[1]==(*omega_ptr)(1));
  REQUIRE(omega[2]==(*omega_ptr)(2));

  // Check Galerkin
  FullMatrix<double> result(n_dir,n_dir);
  FullMatrix<double> const* const M2D(quad.get_M2D());
  FullMatrix<double> const* const D2M(quad.get_D2M());
  D2M->mmult(result,*M2D);
  for (unsigned int i=0; i<n_dir; ++i)
    for (unsigned int j=0; j<n_dir; ++j)
    {
      if (i==j)
        REQUIRE(std::fabs(result(i,j)-1.)<1e-12);
      else
        REQUIRE(std::fabs(result(i,j))<1e-12);
    }
}

TEST_CASE("RTQuadrature/GLC","Check GLC quadrature")
{
  unsigned int n_dir(12);
  unsigned int n_mom(15);
  const double four_pi(4.*M_PI);
  std::vector<double> omega(3,0.);
  omega[0] = 0.868846143426105;
  omega[1] = 0.35988785622265201;
  omega[2] = 0.33998104358485631;

  GLC quad(4,4,false);
  quad.build_quadrature(four_pi,2);

  // Check the number of direction
  REQUIRE(n_dir==quad.get_n_dir());

  // Check the number of moments
  REQUIRE(n_mom==quad.get_n_mom());

  // Check the degree of expansion
  REQUIRE(quad.get_l(0)==0);
  REQUIRE(quad.get_l(1)==1);
  REQUIRE(quad.get_l(2)==1);
  REQUIRE(quad.get_l(3)==2);
  REQUIRE(quad.get_l(4)==2);
  REQUIRE(quad.get_l(5)==2);
  REQUIRE(quad.get_l(6)==3);
  REQUIRE(quad.get_l(7)==3);
  REQUIRE(quad.get_l(8)==3);
  REQUIRE(quad.get_l(9)==3);
  REQUIRE(quad.get_l(10)==4);
  REQUIRE(quad.get_l(11)==4);
  REQUIRE(quad.get_l(12)==4);
  REQUIRE(quad.get_l(13)==4);
  REQUIRE(quad.get_l(14)==4);

  // Check omega
  Vector<double> const* const omega_ptr(quad.get_omega(0));
  REQUIRE(std::fabs(omega[0]-(*omega_ptr)(0))<1e-12);
  REQUIRE(std::fabs(omega[1]-(*omega_ptr)(1))<1e-12);
  REQUIRE(std::fabs(omega[2]-(*omega_ptr)(2))<1e-12);
}
