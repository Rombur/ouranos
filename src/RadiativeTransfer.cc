/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "RadiativeTransfer.hh"

template <int dim,int tensor_dim>
RadiativeTransfer<dim,tensor_dim>::RadiativeTransfer(
    parallel::distributed::Triangulation<dim>* triangulation,
    DoFHandler<dim>* dof_handler,Parameters* parameters,RTQuadrature* quad) :
  triangulation(triangulation),
  dof_handler(dof_handler),
  parameters(parameters),
  quad(quad)
{}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::setup()
{
  // Build the FECells.
  typename DoFHandler<dim>::active_cell_iterator cell(dof_handler->begin_active()),
           end_cell(dof_handler->end());
  const unsigned int fe_order(parameters->get_fe_order());
  FE_DGQ<dim> fe(fe_order);
  dof_handler->distribute_dofs(fe);
  QGauss<dim> quadrature_formula(fe_order+1);
  QGauss<dim-1> face_quadrature_formula(fe_order+1);
  const unsigned int n_quad_points(quadrature_formula.size());
  const unsigned int n_face_quad_points(face_quadrature_formula.size());
  FEValues<dim> fe_values(fe,quadrature_formula,
      update_values|update_gradients|update_JxW_values);
  FEFaceValues<dim> fe_face_values(fe,face_quadrature_formula,
      update_values|update_gradients|update_JxW_values);
  FEFaceValues<dim> fe_neighbor_face_values(fe,face_quadrature_formula,
      update_values);

  for (; cell<end_cell; ++cell)
    if (cell->is_locally_owned())
    {
      FECell<dim,tensor_dim> fecell(n_quad_points,n_face_quad_points,
          fe_values,fe_face_values,fe_neighbor_face_values,cell,end_cell);
      fecell_mesh.push_back(fecell);
    }
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::compute_sweep_ordering()
{
  // First find the cells on the ''boundary'' of the processor
  std::vector<std::set<unsigned int> > boundary_cells(dim);
  const unsigned int fec_mesh_size(fecell_mesh.size());
  for (unsigned int i=0; i<fec_mesh_size; ++i)
  {
    typename DoFHandler<dim>::active_cell_iterator cell(fecell_mesh[i].get_cell());
    for (unsigned int face=0; face<2*dim; ++face)
    {
      // Check if the face is on the boundary of the problem
      if (cell->at_boundary(face)==true)
        boundary_cells[face].insert(i);
      else
        if (cell->neighbor(face)->is_locally_owned()==false)
          boundary_cells[face].insert(i);
    }
  }         

  const unsigned int n_dir(quad->get_n_dir());
  sweep_order.resize(n_dir);
  for (unsigned int idir=0; idir<n_dir; ++idir)
  {
  }
}


//*****Explicit instantiations*****//
template class RadiativeTransfer<2,4>;
template class RadiativeTransfer<2,9>;
template class RadiativeTransfer<2,16>;
template class RadiativeTransfer<2,25>;
template class RadiativeTransfer<2,36>;
template class RadiativeTransfer<3,8>;
template class RadiativeTransfer<3,27>;
template class RadiativeTransfer<3,64>;
template class RadiativeTransfer<3,125>;
template class RadiativeTransfer<3,216>;
