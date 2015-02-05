/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <cmath>
#include <vector>
#include "deal.II/base/point.h"
#include "deal.II/base/quadrature_lib.h"
#include "deal.II/base/tensor.h"
#include "deal.II/fe/fe_dgq.h"
#include "deal.II/fe/fe_values.h"
#include "deal.II/grid/grid_generator.h"
#include "deal.II/grid/tria.h"
#include "../src/FECell.hh"


TEST_CASE("Check FECell on one cell","[FECell][one cell]")
{
  Triangulation<2> triangulation;
  FE_DGQ<2> fe(1);
  DoFHandler<2> dof_handler(triangulation);
  GridGenerator::hyper_cube(triangulation,0,1);
  dof_handler.distribute_dofs(fe);
  QGauss<2> quadrature_formula(2);
  QGauss<1> face_quadrature_formula(2);
  FEValues<2> fe_values(fe,quadrature_formula,
      update_values|update_gradients|update_JxW_values);
  FEFaceValues<2> fe_face_values(fe,face_quadrature_formula,
      update_values|update_gradients|update_normal_vectors|update_JxW_values);
  FEFaceValues<2> fe_neighbor_face_values(fe,face_quadrature_formula,
      update_values);
  DoFHandler<2>::active_cell_iterator cell(dof_handler.begin_active());

  // Create the fecell
  FECell<2,4> fecell(quadrature_formula.size(),face_quadrature_formula.size(),
      fe_values,fe_face_values,fe_neighbor_face_values,cell);

  // Check material_id
  REQUIRE(fecell.get_material_id()==0);

  // Check source_id
  REQUIRE(fecell.get_source_id()==0);

  // Check normal vectors
  Point<2> const * const normal_0 = fecell.get_normal_vector(0);
  REQUIRE(std::fabs((*normal_0)[0]-(-1.))<1e-12);
  REQUIRE(std::fabs((*normal_0)[1])<1e-12);
  Point<2> const * const normal_1 = fecell.get_normal_vector(1);
  REQUIRE(std::fabs((*normal_1)[0]-1.)<1e-12);
  REQUIRE(std::fabs((*normal_1)[1])<1e-12);
  Point<2> const * const normal_2 = fecell.get_normal_vector(2);
  REQUIRE(std::fabs((*normal_2)[0])<1e-12);
  REQUIRE(std::fabs((*normal_2)[1]-(-1.))<1e-12);
  Point<2> const * const normal_3 = fecell.get_normal_vector(3);
  REQUIRE(std::fabs((*normal_3)[0])<1e-12);
  REQUIRE(std::fabs((*normal_3)[1]-1.)<1e-12);
  
  // Check the mass matrix
  Tensor<2,4> mass_matrix(*fecell.get_mass_matrix());
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
  Tensor<2,4> x_grad_matrix(*fecell.get_grad_matrix(0));
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
  
  Tensor<2,4> y_grad_matrix(*fecell.get_grad_matrix(1));
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

  // Check the downwind matrices
  // Left face
  Tensor<2,4> upwind_matrix(*fecell.get_downwind_matrix(0));
  REQUIRE(std::fabs(upwind_matrix[0][0]-(1./3.))<1e-12);
  REQUIRE(upwind_matrix[0][1]==0.);
  REQUIRE(std::fabs(upwind_matrix[0][2]-(1./6.))<1e-12);
  REQUIRE(upwind_matrix[0][3]==0.);
  REQUIRE(std::fabs(upwind_matrix[2][0]-(1./6.))<1e-12);
  REQUIRE(upwind_matrix[2][1]==0.);
  REQUIRE(std::fabs(upwind_matrix[2][2]-(1./3.))<1e-12);
  REQUIRE(upwind_matrix[2][3]==0.);
  for (unsigned int i=0; i<4; ++i)
  {
    REQUIRE(upwind_matrix[1][i]==0.);
    REQUIRE(upwind_matrix[3][i]==0.);
  }
  // Right face
  upwind_matrix = *fecell.get_downwind_matrix(1);
  REQUIRE(upwind_matrix[1][0]==0.);
  REQUIRE(std::fabs(upwind_matrix[1][1]-(1./3.))<1e-12);
  REQUIRE(upwind_matrix[1][2]==0.);
  REQUIRE(std::fabs(upwind_matrix[1][3]-(1./6.))<1e-12);
  REQUIRE(upwind_matrix[3][0]==0.);
  REQUIRE(std::fabs(upwind_matrix[3][1]-(1./6.))<1e-12);
  REQUIRE(upwind_matrix[3][2]==0.);
  REQUIRE(std::fabs(upwind_matrix[3][3]-(1./3.))<1e-12);
  for (unsigned int i=0; i<4; ++i)
  {
    REQUIRE(upwind_matrix[0][i]==0.);
    REQUIRE(upwind_matrix[2][i]==0.);
  }
  // Bottom face
  upwind_matrix = *fecell.get_downwind_matrix(2);
  REQUIRE(std::fabs(upwind_matrix[0][0]-(1./3.))<1e-12);
  REQUIRE(std::fabs(upwind_matrix[0][1]-(1./6.))<1e-12);
  REQUIRE(upwind_matrix[0][2]==0.);
  REQUIRE(upwind_matrix[0][3]==0.);
  REQUIRE(std::fabs(upwind_matrix[1][0]-(1./6.))<1e-12);
  REQUIRE(std::fabs(upwind_matrix[1][1]-(1./3.))<1e-12);
  REQUIRE(upwind_matrix[1][2]==0.);
  REQUIRE(upwind_matrix[1][3]==0.);
  for (unsigned int i=0; i<4; ++i)
  {
    REQUIRE(upwind_matrix[2][i]==0.);
    REQUIRE(upwind_matrix[3][i]==0.);
  }
  // Top face
  upwind_matrix = *fecell.get_downwind_matrix(3);
  REQUIRE(upwind_matrix[2][0]==0.);
  REQUIRE(upwind_matrix[2][1]==0.);
  REQUIRE(std::fabs(upwind_matrix[2][2]-(1./3.))<1e-12);
  REQUIRE(std::fabs(upwind_matrix[2][3]-(1./6.))<1e-12);
  REQUIRE(upwind_matrix[3][0]==0.);
  REQUIRE(upwind_matrix[3][1]==0.);
  REQUIRE(std::fabs(upwind_matrix[3][2]-(1./6.))<1e-12);
  REQUIRE(std::fabs(upwind_matrix[3][3]-(1./3.))<1e-12);
  for (unsigned int i=0; i<4; ++i)
  {
    REQUIRE(upwind_matrix[0][i]==0.);
    REQUIRE(upwind_matrix[1][i]==0.);
  }

  // Check the upwind matrices
  for (unsigned int face=0; face<4; ++face)
  {
    Tensor<2,4> upwind_matrix(*fecell.get_upwind_matrix(face));
    for (unsigned int i=0; i<4; ++i)
      for(unsigned int j=0; j<4; ++j)
        REQUIRE(upwind_matrix[i][j]==0.);
  }
}

TEST_CASE("Check FECell on multiple cells triangulation","[FECELL][multiple cells]")
{
  Triangulation<2> triangulation;
  FE_DGQ<2> fe(1);
  DoFHandler<2> dof_handler(triangulation);
  GridGenerator::subdivided_hyper_cube(triangulation,3,0,3);
  dof_handler.distribute_dofs(fe);
  QGauss<2> quadrature_formula(2);
  QGauss<1> face_quadrature_formula(2);
  FEValues<2> fe_values(fe,quadrature_formula,
      update_values|update_gradients|update_JxW_values);
  FEFaceValues<2> fe_face_values(fe,face_quadrature_formula,
      update_values|update_gradients|update_normal_vectors|update_JxW_values);
  FEFaceValues<2> fe_neighbor_face_values(fe,face_quadrature_formula,
      update_values);
  // Middle cell
  DoFHandler<2>::active_cell_iterator cell(dof_handler.begin_active());
  for (unsigned int i=0; i<4; ++i)
    ++cell;

  // Create the fecell
  FECell<2,4> fecell(quadrature_formula.size(),face_quadrature_formula.size(),
      fe_values,fe_face_values,fe_neighbor_face_values,cell);

  // Check material_id
  REQUIRE(fecell.get_material_id()==0);

  // Check source_id
  REQUIRE(fecell.get_source_id()==0);

  // Check normal vectors
  Point<2> const * const normal_0 = fecell.get_normal_vector(0);
  REQUIRE(std::fabs((*normal_0)[0]-(-1.))<1e-12);
  REQUIRE(std::fabs((*normal_0)[1])<1e-12);
  Point<2> const * const normal_1 = fecell.get_normal_vector(1);
  REQUIRE(std::fabs((*normal_1)[0]-1.)<1e-12);
  REQUIRE(std::fabs((*normal_1)[1])<1e-12);
  Point<2> const * const normal_2 = fecell.get_normal_vector(2);
  REQUIRE(std::fabs((*normal_2)[0])<1e-12);
  REQUIRE(std::fabs((*normal_2)[1]-(-1.))<1e-12);
  Point<2> const * const normal_3 = fecell.get_normal_vector(3);
  REQUIRE(std::fabs((*normal_3)[0])<1e-12);
  REQUIRE(std::fabs((*normal_3)[1]-1.)<1e-12);
  
  // Check the upwind matrices
  // Left face
  Tensor<2,4> upwind_matrix(*fecell.get_upwind_matrix(0));
  REQUIRE(upwind_matrix[0][0]==0.);
  REQUIRE(std::fabs(upwind_matrix[0][1]-(1./3))<1e-12);
  REQUIRE(upwind_matrix[0][2]==0.);
  REQUIRE(std::fabs(upwind_matrix[0][3]-(1./6))<1e-12);
  REQUIRE(std::fabs(upwind_matrix[2][1]-(1./6.))<1e-12);
  REQUIRE(upwind_matrix[2][2]==0.);
  REQUIRE(std::fabs(upwind_matrix[2][3]-(1./3.))<1e-12);
  for (unsigned int i=0; i<4; ++i)
  {
    REQUIRE(upwind_matrix[1][i]==0.);
    REQUIRE(upwind_matrix[3][i]==0.);
  }
  // Right face
  upwind_matrix = *fecell.get_upwind_matrix(1);
  REQUIRE(std::fabs(upwind_matrix[1][0]-(1./3.))<1e-12);
  REQUIRE(upwind_matrix[1][1]==0.);
  REQUIRE(std::fabs(upwind_matrix[1][2]-(1./6.))<1e-12);
  REQUIRE(upwind_matrix[1][3]==0.);
  REQUIRE(std::fabs(upwind_matrix[3][0]-(1./6.))<1e-12);
  REQUIRE(upwind_matrix[3][1]==0.);
  REQUIRE(std::fabs(upwind_matrix[3][2]-(1./3.))<1e-12);
  REQUIRE(upwind_matrix[3][3]==0.);
  for (unsigned int i=0; i<4; ++i)
  {
    REQUIRE(upwind_matrix[0][i]==0.);
    REQUIRE(upwind_matrix[2][i]==0.);
  }
  // Bottom face
  upwind_matrix = *fecell.get_upwind_matrix(2);
  REQUIRE(upwind_matrix[0][0]==0.);
  REQUIRE(upwind_matrix[0][1]==0.);
  REQUIRE(std::fabs(upwind_matrix[0][2]-(1./3.))<1e-12);
  REQUIRE(std::fabs(upwind_matrix[0][3]-(1./6.))<1e-12);
  REQUIRE(upwind_matrix[1][0]==0.);
  REQUIRE(upwind_matrix[1][1]==0.);
  REQUIRE(std::fabs(upwind_matrix[1][2]-(1./6.))<1e-12);
  REQUIRE(std::fabs(upwind_matrix[1][3]-(1./3.))<1e-12);
  for (unsigned int i=0; i<4; ++i)
  {
    REQUIRE(upwind_matrix[2][i]==0.);
    REQUIRE(upwind_matrix[3][i]==0.);
  }
  // Top face
  upwind_matrix = *fecell.get_upwind_matrix(3);
  REQUIRE(std::fabs(upwind_matrix[2][0]-(1./3.))<1e-12);
  REQUIRE(std::fabs(upwind_matrix[2][1]-(1./6.))<1e-12);
  REQUIRE(upwind_matrix[2][2]==0.);
  REQUIRE(upwind_matrix[2][3]==0.);
  REQUIRE(std::fabs(upwind_matrix[3][0]-(1./6.))<1e-12);
  REQUIRE(std::fabs(upwind_matrix[3][1]-(1./3.))<1e-12);
  REQUIRE(upwind_matrix[3][2]==0.);
  REQUIRE(upwind_matrix[3][3]==0.);
  for (unsigned int i=0; i<4; ++i)
  {
    REQUIRE(upwind_matrix[0][i]==0.);
    REQUIRE(upwind_matrix[1][i]==0.);
  }
}
