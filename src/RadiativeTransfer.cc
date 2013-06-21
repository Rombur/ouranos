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
    DoFHandler<dim>* dof_handler,Parameters* parameters,RTQuadrature* quad,
    RTMaterialProperties* material_properties) :
  triangulation(triangulation),
  dof_handler(dof_handler),
  parameters(parameters),
  quad(quad),
  material_properties(material_properties)
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
    // Cells already in the sweep order
    std::set<typename DoFHandler<dim>::active_cell_iterator> used_cells;
    // Candidate cells for the sweep order
    std::list<unsigned int> candidate_cells;
    
    // Find the cells on the boundary
    std::vector<double> boundary_face;
    Vector<double> const* const omega(quad->get_omega(idir)); 
    for (unsigned int i=0; i<dim; ++i)
    {
      if ((*omega)[i]>0.)
        boundary_face.push_back(2*i);
      else
        boundary_face.push_back(2*i+1);
    }
    candidate_cells.resize(boundary_cells[boundary_face[0]].size()+
        boundary_cells[boundary_face[1]].size());
    std::list<unsigned int>::iterator list_it;
    list_it = std::set_union(boundary_cells[boundary_face[0]].begin(),
        boundary_cells[boundary_face[0]].end(),
        boundary_cells[boundary_face[1]].begin(),
        boundary_cells[boundary_face[1]].end(),candidate_cells.begin());
    // Resize candidate_cells
    unsigned int new_size(1);
    std::list<unsigned int>::iterator tmp_it(candidate_cells.begin());
    for (; tmp_it!=list_it; ++new_size)
      ++new_size;
    candidate_cells.resize(new_size);
    if (dim==3)
    {
      std::list<unsigned int> tmp_list(candidate_cells);
      candidate_cells.resize(tmp_list.size()+boundary_cells[boundary_face[2]].size());
      list_it = std::set_union(tmp_list.begin(),tmp_list.end(),
          boundary_cells[boundary_face[2]].begin(),
          boundary_cells[boundary_face[2]].end(),candidate_cells.begin());
      // Resize candidate_cells
      new_size = 1;
      tmp_it = candidate_cells.begin();
      for (; tmp_it!=list_it; ++new_size)
        ++new_size;
      candidate_cells.resize(new_size);
    }

    // Build the sweep order
    while (candidate_cells.size()!=0)
    {
      bool accept(true);
      typename DoFHandler<dim>::active_cell_iterator current_cell(
          fecell_mesh[candidate_cells.front()].get_cell());
      for (unsigned int i=0; i<dim; ++i)
      {
        typename DoFHandler<dim>::active_cell_iterator neighbor_cell(
            current_cell->neighbor(boundary_face[i]));
        // If the upwind neighbors of the current cells are not in the sweep
        // order yet, the cell is rejected
        if ((neighbor_cell->is_locally_owned()==true) &&
            (used_cells.count(neighbor_cell)==0))
          accept = false;
      }
      if (accept==true)
      {   
        // The cell is added to the sweep order
        sweep_order[idir].push_back(candidate_cells.front());
        // The cell is removed from candidate_cells and added to used_cells
        used_cells.insert(fecell_mesh[candidate_cells.front()].get_cell());
        candidate_cells.pop_back();
      }
      else
      {
        // The cell is put at the end of the list.
        unsigned int tmp(candidate_cells.front());
        candidate_cells.pop_front();
        candidate_cells.push_back(tmp);
      }
    }
  }
}

template <int dim,int tensor_dim>
int RadiativeTransfer<dim,tensor_dim>::Apply(Epetra_MultiVector const &x,
    Epetra_MultiVector &y) const
{
  y = x;
  Epetra_MultiVector z(y);

  // Compute the scattering source
  compute_scattering_source(y);
  sweep(y);

  for (int i=0; i<y.MyLength(); ++i)
    y[0][i] = z[0][i]-y[0][i];

  return 0;
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::compute_scattering_source(
    Epetra_MultiVector const &x) const
{
  const unsigned int n_mom(quad->get_n_mom());
  // Reinitialize the scattering source
  for (unsigned int i=0; i<n_mom; ++i)
    (*scattering_source[i]) = 0.;

  typedef typename std::vector<FECell<dim,tensor_dim> >::const_iterator fecell_it;
  fecell_it fecell(fecell_mesh.cbegin());
  fecell_it end_fecell(fecell_mesh.cend());
  unsigned int offset(0);
  Tensor<1,tensor_dim> x_cell;
  for (; fecell!=end_fecell; ++fecell)
  {
    for (unsigned int i=0; i<tensor_dim; ++i)
      x_cell[i] = x[0][offset+i];
    
    Tensor<1,tensor_dim> scat_src_cell((*(fecell->get_mass_matrix()))*x_cell);
    for (unsigned int j=0; j<n_mom; ++j)
    {
      //scat_src_cell *= material_properties->get_sigma_s(fecell->get_material_id(),j);

      for (unsigned int i=0; i<tensor_dim; ++i)
        (*scattering_source[j])[i+offset] += scat_src_cell[i];
    }

    offset += tensor_dim;
  }
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::sweep(Epetra_MultiVector &flux_moments,
    bool rhs) const
{}


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
