/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <string>
#include "../src/RTMaterialProperties.hh"


TEST_CASE("Check RTMaterialProperties","[RTMaterialProperties]")
{
  std::string filename("rtmaterial_properties.inp");
  RTMaterialProperties mat_prop(filename,2,3);

  // Check the number of materials
  REQUIRE(mat_prop.get_n_materials()==2);
  
  // Check the number of groups
  REQUIRE(mat_prop.get_n_groups()==3);

  // Check L_max
  REQUIRE(mat_prop.get_L_max()==0);

  // Check sigma_t
  REQUIRE(mat_prop.get_sigma_t(0,0)==1.);
  REQUIRE(mat_prop.get_sigma_t(0,1)==1.5);
  REQUIRE(mat_prop.get_sigma_t(0,2)==2.);
  REQUIRE(mat_prop.get_sigma_t(1,0)==2.);
  REQUIRE(mat_prop.get_sigma_t(1,1)==2.5);
  REQUIRE(mat_prop.get_sigma_t(1,2)==3.);

  // Check sigma_s
  REQUIRE(mat_prop.get_sigma_s(0,0,0,0)==1.);
  REQUIRE(mat_prop.get_sigma_s(0,0,1,0)==0.5);
  REQUIRE(mat_prop.get_sigma_s(0,0,2,0)==0.4);
  REQUIRE(mat_prop.get_sigma_s(0,1,0,0)==0.3);
  REQUIRE(mat_prop.get_sigma_s(0,1,1,0)==2.);
  REQUIRE(mat_prop.get_sigma_s(0,1,2,0)==0.2);
  REQUIRE(mat_prop.get_sigma_s(0,2,0,0)==0.1);
  REQUIRE(mat_prop.get_sigma_s(0,2,1,0)==0.0);
  REQUIRE(mat_prop.get_sigma_s(0,2,2,0)==3.);
  REQUIRE(mat_prop.get_sigma_s(1,0,0,0)==11.);
  REQUIRE(mat_prop.get_sigma_s(1,0,1,0)==10.5);
  REQUIRE(mat_prop.get_sigma_s(1,0,2,0)==10.4);
  REQUIRE(mat_prop.get_sigma_s(1,1,0,0)==10.3);
  REQUIRE(mat_prop.get_sigma_s(1,1,1,0)==12.);
  REQUIRE(mat_prop.get_sigma_s(1,1,2,0)==10.2);
  REQUIRE(mat_prop.get_sigma_s(1,2,0,0)==10.1);
  REQUIRE(mat_prop.get_sigma_s(1,2,1,0)==10.0);
  REQUIRE(mat_prop.get_sigma_s(1,2,2,0)==13.);
}
