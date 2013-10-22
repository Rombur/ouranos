/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include <string>
#include "deal.II/base/utilities.h"
#include "deal.II/base/mpi.h"
#include "deal.II/fe/fe_dgq.h"
#include "../src/Geometry.hh"


TEST_CASE("Geometry/2D","Check Geometry for 2D")
{
  std::string filename("./tests/geometry_2D.inp");
  FE_DGQ<2> fe(1);
  Geometry<2> geometry(filename,fe);
  // Check the number of divisions
  REQUIRE(geometry.get_n_subdivisions(0)==2);
  REQUIRE(geometry.get_n_subdivisions(1)==3);
  // Check the material IDs
  REQUIRE(geometry.get_material_ids(0,0)==0);
  REQUIRE(geometry.get_material_ids(1,0)==1);
  REQUIRE(geometry.get_material_ids(0,1)==0);
  REQUIRE(geometry.get_material_ids(1,1)==1);
  REQUIRE(geometry.get_material_ids(0,2)==0);
  REQUIRE(geometry.get_material_ids(1,2)==1);
  // Check the number of materials
  REQUIRE(geometry.get_n_materials()==2);
  // Check the source IDs
  REQUIRE(geometry.get_source_ids(0,0)==1);
  REQUIRE(geometry.get_source_ids(1,0)==0);
  REQUIRE(geometry.get_source_ids(0,1)==1);
  REQUIRE(geometry.get_source_ids(1,1)==0);
  REQUIRE(geometry.get_source_ids(0,2)==1);
  REQUIRE(geometry.get_source_ids(1,2)==0);
}

TEST_CASE("Geometry/3D","Check Geometry for 3D")
{
  std::string filename("./tests/geometry_3D.inp");
  FE_DGQ<3> fe(1);
  Geometry<3> geometry(filename,fe);
  // Check the number of divisions
  REQUIRE(geometry.get_n_subdivisions(0)==2);
  REQUIRE(geometry.get_n_subdivisions(1)==3);
  REQUIRE(geometry.get_n_subdivisions(2)==2);
  // Check the material IDs
  REQUIRE(geometry.get_material_ids(0,0,0)==0);
  REQUIRE(geometry.get_material_ids(1,0,0)==1);
  REQUIRE(geometry.get_material_ids(0,1,0)==0);
  REQUIRE(geometry.get_material_ids(1,1,0)==1);
  REQUIRE(geometry.get_material_ids(0,2,0)==0);
  REQUIRE(geometry.get_material_ids(1,2,0)==1);
  REQUIRE(geometry.get_material_ids(0,0,1)==1);
  REQUIRE(geometry.get_material_ids(1,0,1)==0);
  REQUIRE(geometry.get_material_ids(0,1,1)==1);
  REQUIRE(geometry.get_material_ids(1,1,1)==0);
  REQUIRE(geometry.get_material_ids(0,2,1)==1);
  REQUIRE(geometry.get_material_ids(1,2,1)==0);
  // Check the number of materials
  REQUIRE(geometry.get_n_materials()==2);
  // Check the source IDs
  REQUIRE(geometry.get_source_ids(0,0,0)==1);
  REQUIRE(geometry.get_source_ids(1,0,0)==0);
  REQUIRE(geometry.get_source_ids(0,1,0)==1);
  REQUIRE(geometry.get_source_ids(1,1,0)==0);
  REQUIRE(geometry.get_source_ids(0,2,0)==1);
  REQUIRE(geometry.get_source_ids(1,2,0)==0);
  REQUIRE(geometry.get_source_ids(0,0,1)==2);
  REQUIRE(geometry.get_source_ids(1,0,1)==1);
  REQUIRE(geometry.get_source_ids(0,1,1)==2);
  REQUIRE(geometry.get_source_ids(1,1,1)==1);
  REQUIRE(geometry.get_source_ids(0,2,1)==2);
  REQUIRE(geometry.get_source_ids(1,2,1)==1);
}

int main (int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv);

  int result = Catch::Main(argc,argv);

  return result;
}
