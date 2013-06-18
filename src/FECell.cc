/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "FECell.hh"


template <int dim,int tensor_dim>
FECell<dim,tensor_dim>::FECell(const unsigned int n_q_points,
    const unsigned int n_face_q_points,FEValues<dim> &fe_values,
    FEFaceValues<dim> &fe_face_values,FEFaceValues<dim> &fe_neighbor_face_values,
    typename DoFHandler<dim>::active_cell_iterator &cell,
    typename DoFHandler<dim>::active_cell_iterator &end_cell) :
  cell(cell),
  grad_matrices(dim),
  downwind_matrices(2*dim),
  upwind_matrices(2*dim)
{
  source_id = cell->material_id()/100;
  material_id = cell->material_id()-100*source_id;
  unsigned int face_map[6] = {1,0,3,2,6,5};

  // Reinit fe_values on the current cell
  fe_values.reinit(cell);

  // Build the mass matrix
  for (unsigned int i=0; i<tensor_dim; ++i)
    for (unsigned int j=0; j<tensor_dim; ++j)
      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
        mass_matrix[i][j] += fe_values.shape_value(i,q_point)*
          fe_values.shape_value(j,q_point)*fe_values.JxW(q_point);

  // Build the gradient matrices
  for (unsigned int d=0; d<dim; ++d)
    for (unsigned int i=0; i<tensor_dim; ++i)
      for (unsigned int j=0; j<tensor_dim; ++j)
        for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
          grad_matrices[d][i][j] += fe_values.shape_value(j,q_point)*
            fe_values.shape_grad(i,q_point)[d]*fe_values.JxW(q_point);

  // Loop over the faces to create the downwind matrices
  for (unsigned int face=0; face<2*dim; ++face)
  {
    // Reinit fe_face_values on the current face
    fe_face_values.reinit(cell,face);
    for (unsigned int i=0; i<tensor_dim; ++i)
      for (unsigned int j=0; j<tensor_dim; ++j)
        for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
          downwind_matrices[face][i][j] += fe_face_values.shape_value(i,q_point)*
            fe_face_values.shape_value(j,q_point)*fe_face_values.JxW(q_point);
  }

  // Loop over the faces to create the upwind matrices
  typename DoFHandler<dim>::active_cell_iterator neighbor_cell;
  for (unsigned int face=0; face<2*dim; ++face)
  {
    // Reinit fe_face_values on the current face
    fe_face_values.reinit(cell,face);
    neighbor_cell = cell->neighbor(face);
    if (neighbor_cell!=end_cell)
    {
      fe_neighbor_face_values.reinit(neighbor_cell,face_map[face]);
      for (unsigned int i=0; i<tensor_dim; ++i)
        for (unsigned int j=0; j<tensor_dim; ++j)
          for (unsigned int q_point=0; q_point<n_face_q_points; ++q_point)
            upwind_matrices[face][i][j] += fe_face_values.shape_value(i,q_point)*
              fe_neighbor_face_values.shape_value(j,q_point)*
              fe_face_values.JxW(q_point);
    }
  }
}

//*****Explicit instantiations*****//
template class FECell<2,4>;
template class FECell<2,9>;
template class FECell<2,16>;
template class FECell<2,25>;
template class FECell<2,36>;
template class FECell<3,8>;
template class FECell<3,27>;
template class FECell<3,64>;
template class FECell<3,125>;
template class FECell<3,216>;
