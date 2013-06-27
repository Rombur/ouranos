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
    RTMaterialProperties* material_properties,Epetra_Comm const* comm,
    Epetra_Map const* map) :
  n_mom(quad->get_n_mom()),
  group(0),
  n_groups(1),
  n_cells(1),
  comm(comm),
  map(map),
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
      update_values|update_gradients|update_normal_vectors|update_JxW_values);
  FEFaceValues<dim> fe_neighbor_face_values(fe,face_quadrature_formula,
      update_values);

  for (; cell<end_cell; ++cell)
    if (cell->is_locally_owned())
    {
      FECell<dim,tensor_dim> fecell(n_quad_points,n_face_quad_points,
          fe_values,fe_face_values,fe_neighbor_face_values,cell,end_cell);
      fecell_mesh.push_back(fecell);
    }
  n_cells = fecell_mesh.size();
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::compute_sweep_ordering()
{
  // First find the cells on the ''boundary'' of the processor
  std::vector<std::set<unsigned int> > boundary_cells(dim);
  const unsigned int fec_mesh_size(fecell_mesh.size());
  for (unsigned int i=0; i<fec_mesh_size; ++i)
  {
    typename DoFHandler<dim>::active_cell_iterator cell(*fecell_mesh[i].get_cell());
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
          *fecell_mesh[candidate_cells.front()].get_cell());
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
        used_cells.insert(current_cell);
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

// DO SOMETHING DIFFERENT HERE
  // Clear flux_moments
  y.PutScalar(0.);
  const unsigned int n_dir(quad->get_n_dir());
  for (unsigned int idir=0; idir<n_dir; ++idir)
    sweep(y,idir);

  for (int i=0; i<y.MyLength(); ++i)
    y[0][i] = z[0][i]-y[0][i];

  return 0;
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::compute_scattering_source(
    Epetra_MultiVector const &x) const
{
  // Reinitialize the scattering source
  for (unsigned int i=0; i<n_mom; ++i)
    (*scattering_source[i]) = 0.;

  typedef typename std::vector<FECell<dim,tensor_dim> >::const_iterator fecell_it;
  fecell_it fecell(fecell_mesh.cbegin());
  fecell_it end_fecell(fecell_mesh.cend());
  Tensor<1,tensor_dim> x_cell;
  std::vector<int> local_dof_indices(tensor_dim);
  for (; fecell!=end_fecell; ++fecell)
  {
    get_multivector_indices(local_dof_indices,*fecell->get_cell());
    for (unsigned int i=0; i<tensor_dim; ++i)
      x_cell[i] = x[0][local_dof_indices[i]];
    
    Tensor<1,tensor_dim> scat_src_cell((*(fecell->get_mass_matrix()))*x_cell);
    for (unsigned int j=0; j<n_mom; ++j)
    {
      scat_src_cell *= material_properties->get_sigma_s(fecell->get_material_id(),
          group,group,j);

      for (unsigned int i=0; i<tensor_dim; ++i)
        (*scattering_source[j])[local_dof_indices[i]] += scat_src_cell[i];
    }
  }
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::compute_outer_scattering_source( 
    Tensor<1,tensor_dim> &b,Epetra_MultiVector const* const group_flux,
    FECell<dim,tensor_dim> const* const fecell,const unsigned int idir) const
{
  const unsigned int n_local_dofs(dof_handler->n_locally_owned_dofs());
  FullMatrix<double> const* const M2D(quad->get_M2D());
  Tensor<1,tensor_dim> x_cell;
  std::vector<int> local_dof_indices(tensor_dim);
  get_multivector_indices(local_dof_indices,*fecell->get_cell());
  for (unsigned int g=0; g<n_groups; ++g)
  {
    if (g!=group)
    {
      for (unsigned int i=0; i<n_mom; ++i)
      {
        double m2d((*M2D)(idir,i));
        for (unsigned int j=0; j<tensor_dim; ++j)
          x_cell[j] = (*group_flux)[g][i*n_local_dofs+local_dof_indices[j]];

        Tensor<1,tensor_dim> scat_src_cell((*(fecell->get_mass_matrix()))*x_cell);

        scat_src_cell *= (m2d*material_properties->get_sigma_s(
              fecell->get_material_id(),g,group,i));

        b += scat_src_cell;
      }
    }
  }
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::sweep(Epetra_MultiVector &flux_moments,
    unsigned int idir,Epetra_MultiVector const* const group_flux) const
{
  FullMatrix<double> const* const M2D(quad->get_M2D());
  FullMatrix<double> const* const D2M(quad->get_D2M());
  Epetra_MultiVector psi(*map,1);
  Vector<double> const* const omega(quad->get_omega(idir));
  std::vector<int> local_dof_indices(tensor_dim);

  // Sweep on the spatial cells
  for (unsigned int i=0; i<n_cells; ++i)
  {
    FECell<dim,tensor_dim> const* const fecell = &fecell_mesh[sweep_order[idir][i]];
    typename DoFHandler<dim>::active_cell_iterator const cell(*fecell->get_cell());
    Tensor<1,tensor_dim> b;
    Tensor<2,tensor_dim> A(*(fecell->get_mass_matrix()));
    get_multivector_indices(local_dof_indices,cell);
    // Volumetrix terms of the lhs: -omega dot grad_matrix + sigma_t mass
    A *= material_properties->get_sigma_t(fecell_mesh[i].get_material_id(),group);
    for (unsigned int d=0; d<dim; ++d)
      A += (-(*omega)[i]*(*(fecell->get_grad_matrix(d))));
    
    // Scattering source
    for (unsigned int mom=0; mom<n_mom; ++mom)
    {
      const double m2d((*M2D)(idir,mom));
      for (unsigned int j=0; j<tensor_dim; ++i)
        b[j] += m2d*(*scattering_source[mom])[local_dof_indices[j]];
    }
    if (group_flux!=nullptr)
    {
      // Divide the source by the sum of the weights to the input source is
      // easier to set
      Tensor<1,tensor_dim> src(parameters->get_src(fecell->get_source_id(),group));
      b += (*fecell->get_mass_matrix())*src;
      // Compute the scattering source due to the other groups
      compute_outer_scattering_source(b,group_flux,fecell,idir);
    }

    // Surfacic terms
    for (unsigned int face=0; face<2*dim; ++face)
    {
      Point<dim> const* const normal_vector = fecell->get_normal_vector(face);
      double n_dot_omega(0.);
      for (unsigned int j=0; j<dim; ++j)
        n_dot_omega += (*omega)[j]*(*normal_vector)(i);

      if (n_dot_omega<0.)
      {
        // Upwind
        if (cell->at_boundary(face)==false)
        {
          Tensor<2,tensor_dim> const* const upwind_matrix(
              fecell->get_upwind_matrix(face));
          Tensor<1,tensor_dim> psi_cell(-n_dot_omega);
          typename DoFHandler<dim>::active_cell_iterator neighbor_cell;
          neighbor_cell = cell->neighbor(face);
          if (neighbor_cell->is_locally_owned()==true)
          {
            std::vector<int> neighbor_local_dof_indices(tensor_dim);
            get_multivector_indices(neighbor_local_dof_indices,neighbor_cell);
            for (unsigned int j=0; j<tensor_dim; ++j)
              psi_cell[j] *= psi[0][neighbor_local_dof_indices[j]];
          }
          else
          {
// DO STH DIFFERENT HERE          
          }

          b += (*upwind_matrix)*psi_cell;
        }
        else
        {
          if (group_flux!=nullptr)
          {
            double inc_flux_val(0.);
            Tensor<2,tensor_dim> const* const downwind_matrix(
              fecell->get_downwind_matrix(face));
            if ((parameters->get_bc_type(face)==MOST_NORMAL) ||
                (parameters->get_bc_type(face)==ISOTROPIC))
              inc_flux_val = parameters->get_inc_flux(face,group);
            inc_flux_val /= parameters->get_weight_sum();
            Tensor<1,tensor_dim> inc_flux(inc_flux_val);
            b += (-n_dot_omega)*(*downwind_matrix)*inc_flux;
          }
        }
      }
      else
      {
        // Downwind
        A += n_dot_omega*(*fecell->get_downwind_matrix(face));
      }
    }

    // Solve the linear system
    Tensor<1,tensor_dim,unsigned int> pivot;
    Tensor<1,tensor_dim> x;
    LU_decomposition(A,pivot);
    LU_solve(A,b,x,pivot);
   
    for (unsigned int j=0; j<tensor_dim; ++j)
      psi[0][local_dof_indices[j]] = x[j];
  }

  // Update flux moments
  const unsigned int n_local_dofs(dof_handler->n_locally_owned_dofs());
  for (unsigned int mom=0; mom<n_mom; ++mom)
  {
    const double d2m((*D2M)(mom,idir));
    for (unsigned int i=0; i<n_local_dofs; ++i)
      flux_moments[0][i+mom*n_local_dofs] += d2m*psi[0][i];
  }
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::get_multivector_indices(
    std::vector<int> &dof_indices,
    typename DoFHandler<dim>::active_cell_iterator const& cell) const
{
  std::vector<types::global_dof_index> local_dof_indices(tensor_dim);
  cell->get_dof_indices(local_dof_indices);
  for (unsigned int i=0; i<tensor_dim; ++i)
    dof_indices[i] = map->LID(static_cast<TrilinosWrappers::types::int_type>
        (local_dof_indices[i]));
}
  
template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::LU_decomposition(
    Tensor<2,tensor_dim> &A,Tensor<1,tensor_dim,unsigned int> &pivot) const
{
  double max(0.);
  for (unsigned int k=0; k<tensor_dim; ++k)
  {
    // Find the pivot row
    pivot[k] = k;
    max = std::fabs(A[k][k]);
    for (unsigned int j=k+1; j<tensor_dim; ++j)
      if (max<std::fabs(A[j][k]))
      {
        max = std::fabs(A[j][k]);
        pivot[k] = j;
      }
    
    // If the pivot row differs from the current row, then interchange the two
    // rows
    if (pivot[k]!=k)
    {
      const unsigned int piv(pivot[k]);
      for (unsigned int j=0; j<tensor_dim; ++j)
      {
        max = A[k][j];
        A[k][j] = A[piv][j];
        A[piv][j] = max;
      }
    }

    // Find the upper triangular matrix elements for row k
    for (unsigned int j=k+1; j<k; ++j)
      A[k][j] /= A[k][k];

    // Update remaining matrix
    for (unsigned int i=k+1; i<tensor_dim; ++i)
      for (unsigned int j=k+1; j<tensor_dim; ++j)
        A[i][j] -= A[i][k]*A[k][j];
  }
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::LU_solve(Tensor<2,tensor_dim> const &A,
    Tensor<1,tensor_dim> &b,Tensor<1,tensor_dim> &x,
    Tensor<1,tensor_dim,unsigned int> const &pivot) const
{
  // Solve the linear equation \f$Lx=b\f$ for \f$x\f$  where \f$L\f$ is a
  // lower triangular matrix
  for (unsigned int k=0; k<tensor_dim; ++k)
  {
    if (pivot[k]!=k)
    {
      double tmp(b[k]);
      b[k] = b[pivot[k]];
      b[pivot[k]] = tmp;
    }
    x[k] = b[k];
    for (unsigned int i=0; i<tensor_dim; ++i)
      x[k] -= x[i]*A[k][i];
    x[k] /= A[k][k];
  }

  // Solve the linear equation \f$Ux=y\$, where \f$y\f$ is the solution
  // obtained above of \f$Lx=b\f$ and \f$U\f$ is an upper triangular matrix.
  // The elements of the diagonal of the upper triangular part of the matrix are 
  // assumed to be ones.
  // To avoid warning about comparison between unsigned int and int, k is
  // unsigned int. Thus, the condition k>=0 becomes k<max_unsigned_int.
  for (unsigned int k=tensor_dim-1; k<tensor_dim; --k)
  {
    if (pivot[k]!=k)
    {
      double tmp(b[k]);
      b[k] = b[pivot[k]];
      b[pivot[k]] = tmp;
    }
    for (unsigned int i=k+1; i<tensor_dim; ++i)
      x[k] -= x[i]*A[k][i];
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
