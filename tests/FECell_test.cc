/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <cmath>
#include "deal.II/base/quadrature_lib.h"
#include "deal.II/base/tensor.h"
#include "deal.II/fe/fe_dgq.h"
#include "deal.II/fe/fe_values.h"
#include "deal.II/grid/grid_generator.h"
#include "deal.II/grid/tria.h"
#include "../src/FECell.hh"

TEST_CASE("FECell","Check FECell")
{
  Triangulation<2> triangulation;
  FE_DGQ<2> fe(1);
  DoFHandler<2> dof_handler(triangulation);
  GridGenerator::hyper_cube(triangulation,0,1);
  dof_handler.distribute_dofs(fe);
  QGauss<2> quadrature_formula(2);
  FEValues<2> fe_values(fe,quadrature_formula,
      update_values|update_gradients|update_JxW_values);
  DoFHandler<2>::active_cell_iterator cell(dof_handler.begin_active());

  // Create the fecell
  FECell<2,4> fecell(quadrature_formula.size(),fe_values);
  
  // Check the mass matrix
  Tensor<2,4> mass_matrix(fecell.get_mass_matrix());
  for (unsigned int i=0; i<4; ++i)
    REQUIRE(std::fabs(mass_matrix[i][i]-(1./9.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[0][1]-(1./18.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[0][2]-(1./18.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[1][0]-(1./18.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[1][3]-(1./18.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[2][0]-(1./18.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[2][3]-(1./18.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[3][1]-(1./18.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[3][2]-(1./18.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[0][3]-(1./36.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[1][2]-(1./36.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[2][1]-(1./36.))<1e-12);
  REQUIRE(std::fabs(mass_matrix[3][0]-(1./36.))<1e-12);

  // Check the gradient matrices
  const double grad_ratio(1./12.);
  Tensor<2,4> x_grad_matrix(fecell.get_grad_matrix(0));
  REQUIRE(std::fabs(x_grad_matrix[0][0]-(-2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[0][1]-(-2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[0][2]-(-grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[0][3]-(-grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[1][0]-(2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[1][1]-(2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[1][2]-(grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[1][3]-(grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[2][0]-(-grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[2][1]-(-grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[2][2]-(-2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[2][3]-(-2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[3][0]-(grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[3][1]-(grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[3][2]-(2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(x_grad_matrix[3][3]-(2.*grad_ratio))<1e-12);
  
  Tensor<2,4> y_grad_matrix(fecell.get_grad_matrix(1));
  REQUIRE(std::fabs(y_grad_matrix[0][0]-(-2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[0][1]-(-grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[0][3]-(-grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[0][2]-(-2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[1][0]-(-grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[1][1]-(-2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[1][2]-(-grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[1][3]-(-2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[3][0]-(grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[3][1]-(2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[3][2]-(grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[3][3]-(2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[2][0]-(2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[2][1]-(grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[2][2]-(2.*grad_ratio))<1e-12);
  REQUIRE(std::fabs(y_grad_matrix[2][3]-(grad_ratio))<1e-12);
}
