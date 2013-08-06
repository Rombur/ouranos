/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Pleas refer to the file
 * license.txt for the text and further information on this license.
 */

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <string>
#include <vector>
#include "../src/Parameters.hh"


TEST_CASE("Parameters","Check the parameters handler")
{                    
  std::string filename("./tests/parameters.inp");
  Parameters parameters(filename);

  REQUIRE(parameters.get_dimension()==2);
  REQUIRE(parameters.get_fe_order()==4);
  REQUIRE(parameters.get_geometry_filename()=="geometry.inp");
  REQUIRE(parameters.get_refinement_factor()==0.3);
  REQUIRE(parameters.get_coarsening_factor()==0.1);

  REQUIRE(parameters.get_xs_filename()=="xs.inp");
  REQUIRE(parameters.get_solver_type()==BICGSTAB);
  REQUIRE(parameters.get_max_outer_it()==2);
  REQUIRE(parameters.get_max_inner_it()==10);
  REQUIRE(parameters.get_outer_tolerance()==1e-8);
  REQUIRE(parameters.get_inner_tolerance()==1e-10);
  REQUIRE(parameters.get_sn_order()==6);
  REQUIRE(parameters.get_quad_type()==GLC_QUAD);
  REQUIRE(parameters.get_weight_sum()==2.*M_PI);
  REQUIRE(parameters.get_galerkin()==false);
  REQUIRE(parameters.get_bc_type(0)==VACUUM);
  REQUIRE(parameters.get_bc_type(1)==ISOTROPIC);
  REQUIRE(parameters.get_bc_type(2)==REFLECTIVE);
  REQUIRE(parameters.get_bc_type(3)==MOST_NORMAL);
  REQUIRE(parameters.get_inc_flux(1,0)==1.);
  REQUIRE(parameters.get_inc_flux(1,1)==2.);
  REQUIRE(parameters.get_inc_flux(3,0)==8.);
  REQUIRE(parameters.get_inc_flux(3,1)==9.);
  REQUIRE(parameters.get_n_src()==1);
  REQUIRE(parameters.get_n_groups()==2);
  REQUIRE(parameters.get_src(0,0)==4.5);
  REQUIRE(parameters.get_src(0,1)==8.9);
}
