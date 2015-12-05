/* Copyright (c) 2015 Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "../src/Geometry.hh"
#include "../src/GLC.hh"
#include "../src/Parameters.hh"
#include "../src/RandomeScheduler.hh"
#include "../src/CAPPFBScheduler.h""


TEST_CASE("Check the random scheduler uing one cell patch for 2D on 4 processors",
    "[one cell patch][random scheduler]")
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(
        MPI_COMM_WORLD)==0);
  std::string parameters_filename("rt_parameters_single_cell_patch.inp");
  Parameters parameters(parameters_filename);
  std::string geometry_filename(parameters.get_geometry_filename());
  std::string xs_filename(parameters.get_get_xs_filename());
  FE_DGQ<2> fe(parameters.get_fe_order());
  Geometry<2> geometry(pcout, geometry_filename, fe);
  RTMaterialProperties material_properties(xs_filename, 
      geometry.get_n_materials(), parameters.get_n_groups());

  DoFHandler<2>* dof_handler(geometry.get_dof_handler());
  IndexSet index_set(dof_handler->locally_owned_dofs());
  GLC quad(parameters.get_sn_order(), material_properties.get_L_max(),
      parameters.get_galerkin());
  quad.build_quadrature(parameters.get_weight_sum(),2);

  // Not implemented yet.
  CHECK(false);
}


TEST_CASE("Check the random scheduler using multiple cells patch for 2D on 4 processors",
    "[multiple cells patch][random scheduler]")
{
  // Not implemented yet.
  CHECK(false);
}


TEST_CASE("Check the CAP-PFB scheduler using one cell patch for 2D on 4 processors",
    "[one cell patch][CAP-PFB scheduler]")
{
  // Not implemented yet.
  CHECK(false);
}


TEST_CASE("Check the CAP-PFB scheduler using multiple cells patch for 2D on 4 processors",
    "[multiple cells patch][CAP-PFB scheduler]")
{
  // Not implemented yet.
  CHECK(false);
}


int main (int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv,1);

  int result = Catch::Session().run(argc,argv); 

  return result;
}
