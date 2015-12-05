/* Copyright (c) 2013, 2014 Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "RadiativeTransfer.hh"

#include <cmath>
#include <map>
#include "mpi.h"
#include "deal.II/base/point.h"
#include "deal.II/base/quadrature_lib.h"
#include "deal.II/fe/fe_values.h"
#include "deal.II/lac/full_matrix.h"


template <int dim,int tensor_dim>
RadiativeTransfer<dim,tensor_dim>::RadiativeTransfer(unsigned int n_groups,
    types::global_dof_index n_dofs,FE_DGQ<dim>* fe,
    parallel::distributed::Triangulation<dim>* triangulation,
    DoFHandler<dim>* dof_handler,Parameters* parameters,RTQuadrature* quad,
    RTMaterialProperties* material_properties,Epetra_MpiComm const* comm,
    Epetra_Map const* map,std::shared_ptr<Scheduler<dim,tensor_dim>> scheduler) :
  n_mom(quad->get_n_mom()),
  group(0),
  n_groups(n_groups),
  n_dofs(n_dofs),
  comm(comm),
  map(map),
  fe(fe),
  triangulation(triangulation),
  dof_handler(dof_handler),
  parameters(parameters),
  quad(quad),
  material_properties(material_properties),
  scheduler(scheduler)
{
  const unsigned int n_mom(quad->get_n_mom());
  scattering_source.resize(n_mom,nullptr);
  for (unsigned int i=0; i<n_mom; ++i)
    scattering_source[i] = new Vector<double>(dof_handler->n_locally_owned_dofs());
}


template <int dim,int tensor_dim>
RadiativeTransfer<dim,tensor_dim>::~RadiativeTransfer()
{
  for (unsigned int i=0; i<scattering_source.size(); ++i)
    if (scattering_source[i]!=nullptr)
    {
      delete scattering_source[i];
      scattering_source[i] = nullptr;
    }
}


template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::setup()
{
  // Build the FECells
  const unsigned int fe_order(parameters->get_fe_order());
  dof_handler->distribute_dofs(*fe);
  active_cell_iterator cell(dof_handler->begin_active()), end_cell(dof_handler->end());
  QGauss<dim> quadrature_formula(fe_order+1);
  QGauss<dim-1> face_quadrature_formula(fe_order+1);
  const unsigned int n_quad_points(quadrature_formula.size());
  const unsigned int n_face_quad_points(face_quadrature_formula.size());
  FEValues<dim> fe_values(*fe,quadrature_formula,
      update_values|update_gradients|update_JxW_values);
  FEFaceValues<dim> fe_face_values(*fe,face_quadrature_formula,
      update_values|update_gradients|update_normal_vectors|update_JxW_values);
  FEFaceValues<dim> fe_neighbor_face_values(*fe,face_quadrature_formula,
      update_values);

  std::map<active_cell_iterator,unsigned int> cell_to_fecell_map;
  unsigned int fecell_id(0);
  for (; cell<end_cell; ++cell)
    if (cell->is_locally_owned())
    {
      FECell<dim,tensor_dim> fecell(n_quad_points,n_face_quad_points,
          fe_values,fe_face_values,fe_neighbor_face_values,cell);
      fecell_mesh.push_back(fecell);
      cell_to_fecell_map[cell] = fecell_id;
      ++fecell_id;
    }

  // Create the tasks used in the sweep.
  scheduler->setup(parameters->get_n_levels_patch(),&fecell_mesh,cell_to_fecell_map);
}


template <int dim,int tensor_dim>
int RadiativeTransfer<dim,tensor_dim>::Apply(Epetra_MultiVector const &x,
    Epetra_MultiVector &y) const
{
  y = x;
  Epetra_MultiVector z(y);

  // Compute the scattering source
  compute_scattering_source(y);

  // Clear flux_moments
  y.PutScalar(0.);

  // Create the buffers and the MPI_Request
  std::list<double*> buffers;
  std::list<MPI_Request*> requests;
  // Start the scheduler by creating the tasks_ready list
  scheduler->start();
  // Sweep through the mesh
  while (scheduler->get_n_tasks_to_execute()!=0)
  {
    sweep(*(scheduler->get_next_task()),buffers,requests,y);
    scheduler->free_buffers(buffers,requests);
  }
  // Free all the buffers left.
  while (buffers.size()!=0)
    scheduler->free_buffers(buffers,requests);

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

  // Loop over the FECells 
  typedef typename std::vector<FECell<dim,tensor_dim> >::const_iterator fecell_it;
  fecell_it fecell(fecell_mesh.cbegin());
  fecell_it end_fecell(fecell_mesh.cend());
  Tensor<1,tensor_dim> x_cell;
  std::vector<int> local_dof_indices(tensor_dim);
  for (; fecell!=end_fecell; ++fecell)
  {
    get_multivector_indices(local_dof_indices,fecell->get_cell());
    for (unsigned int j=0; j<n_mom; ++j)
    {
      for (unsigned int i=0; i<tensor_dim; ++i)
        x_cell[i] = x[0][j*n_dofs+local_dof_indices[i]];

      Tensor<1,tensor_dim> scat_src_cell((*(fecell->get_mass_matrix()))*x_cell);
      scat_src_cell *= material_properties->get_sigma_s(fecell->get_material_id(),
          group,group,j);

      // scatter_source used multivector indices because multivector indices
      // are just the local indices, i.e, in [0,n_locally_owned_dofs[
      for (unsigned int i=0; i<tensor_dim; ++i)
        (*scattering_source[j])[local_dof_indices[i]] += scat_src_cell[i];
    }
  }
}


template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::compute_outer_scattering_source( 
    Tensor<1,tensor_dim> &b,std::vector<TrilinosWrappers::MPI::Vector> const* const 
    group_flux,FECell<dim,tensor_dim> const* const fecell,const unsigned int idir) 
  const
{
  // Does the same thing that compute_scattering_source but on the other
  // groups
  FullMatrix<double> const* const M2D(quad->get_M2D());
  Tensor<1,tensor_dim> x_cell;
  std::vector<int> local_dof_indices(tensor_dim);
  get_multivector_indices(local_dof_indices,fecell->get_cell());
  for (unsigned int g=0; g<n_groups; ++g)
  {
    if (g!=group)
    {
      for (unsigned int i=0; i<n_mom; ++i)
      {
        double m2d((*M2D)(idir,i));
        for (unsigned int j=0; j<tensor_dim; ++j)
          x_cell[j] = (*group_flux)[g*n_mom+i][local_dof_indices[j]];

        Tensor<1,tensor_dim> scat_src_cell((*(fecell->get_mass_matrix()))*x_cell);

        scat_src_cell *= (m2d*material_properties->get_sigma_s(
              fecell->get_material_id(),g,group,i));

        b += scat_src_cell;
      }
    }
  }
}


template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::sweep(Task const &task,
    std::list<double*> &buffers,std::list<MPI_Request*> &requests,
    Epetra_MultiVector &flux_moments,
    std::vector<TrilinosWrappers::MPI::Vector> const* const group_flux) const
{
  const unsigned int idir(task.get_idir());
  const unsigned int sweep_order_size(task.get_sweep_order_size());
  std::vector<unsigned int> const* sweep_order(task.get_sweep_order());
  FullMatrix<double> const* const M2D(quad->get_M2D());
  FullMatrix<double> const* const D2M(quad->get_D2M());
  Vector<double> const* const omega(quad->get_omega(idir));
  std::vector<int> multivector_indices(tensor_dim);
  std::set<active_cell_iterator> cells_in_task;
  std::unordered_map<types::global_dof_index,double> angular_flux;
 
  // Sweep on the spatial cells of the current task
  for (unsigned int i=0; i<sweep_order_size; ++i)
  {
    FECell<dim,tensor_dim> const* const fecell = &fecell_mesh[(*sweep_order)[i]];
    active_cell_iterator const cell(fecell->get_cell());
    Tensor<1,tensor_dim> b;
    Tensor<2,tensor_dim> A(*(fecell->get_mass_matrix()));
    get_multivector_indices(multivector_indices,cell);
    // Volumetric terms of the lhs: -omega dot grad_matrix + sigma_t mass
    A *= material_properties->get_sigma_t(fecell_mesh[
        (*sweep_order)[i]].get_material_id(),group);
    for (unsigned int d=0; d<dim; ++d)
      A += (-(*omega)[d]*(*(fecell->get_grad_matrix(d))));
    
    // Scattering source
    for (unsigned int mom=0; mom<n_mom; ++mom)
    {
      const double m2d((*M2D)(idir,mom));
      for (unsigned int j=0; j<tensor_dim; ++j)
        b[j] += m2d*(*scattering_source[mom])[multivector_indices[j]];
    }
    if (group_flux!=nullptr)
    {
      // Divide the source by the sum of the weights so the input source is
      // easier to set
      Tensor<1,tensor_dim> src;
      for (unsigned int j=0; j<tensor_dim; ++j)
        src[j] = parameters->get_src(fecell->get_source_id(),group)/
          parameters->get_weight_sum();
      b += (*fecell->get_mass_matrix())*src;
      // Compute the scattering source due to the other groups
      compute_outer_scattering_source(b,group_flux,fecell,idir);
    }

    // Surfacic terms
    for (unsigned int face=0; face<2*dim; ++face)
    {
      Tensor<1,dim> const* normal_vector = fecell->get_normal_vector(face);
      double n_dot_omega(0.);
      for (unsigned int d=0; d<dim; ++d)
        n_dot_omega += (*omega)[d]*(*normal_vector)[d];

      if (n_dot_omega<0.)
      {
        // Upwind
        if (cell->at_boundary(face)==false)
        {
          Tensor<2,tensor_dim> const* const upwind_matrix(
              fecell->get_upwind_matrix(face));
          Tensor<1,tensor_dim> psi_cell;
          for (unsigned int j=0; j<tensor_dim; ++j)
            psi_cell[j] = -n_dot_omega;
          active_cell_iterator neighbor_cell;
          neighbor_cell = cell->neighbor(face);
          std::vector<types::global_dof_index> neighbor_dof_indices(tensor_dim);
          neighbor_cell->get_dof_indices(neighbor_dof_indices);
          // If the neighbor cell is in the current task, get the value from
          // angular_flux. Otherwise, get the value from required_dofs.
          if (cells_in_task.count(neighbor_cell)!=0)
            for (unsigned int j=0; j<tensor_dim; ++j)
              psi_cell[j] *= angular_flux[neighbor_dof_indices[j]];
          else
            for (unsigned int j=0; j<tensor_dim; ++j)
              psi_cell[j] *= task.get_required_angular_flux(neighbor_dof_indices[j]);

          b += (*upwind_matrix)*psi_cell;
        }
        else
        {
          // Use only to build the rhs of GMRES
          if (group_flux!=nullptr)
          {
            double inc_flux_val(0.);
            Tensor<2,tensor_dim> const* const downwind_matrix(
                fecell->get_downwind_matrix(face));
            if (((parameters->get_bc_type(face)==MOST_NORMAL) &&
                  (quad->is_most_normal_direction(face,idir)==true))||
                (parameters->get_bc_type(face)==ISOTROPIC))
              inc_flux_val = parameters->get_inc_flux(face,group);
            inc_flux_val /= parameters->get_weight_sum();
            Tensor<1,tensor_dim> inc_flux;
            for (unsigned int j=0; j<tensor_dim; ++j)
              inc_flux[j] = inc_flux_val;
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

    // Update flux moments
    for (unsigned int mom=0; mom<n_mom; ++mom)
    {
      const double d2m((*D2M)(mom,idir));
      for (unsigned int j=0; j<tensor_dim; ++j)
        flux_moments[0][mom*n_dofs+multivector_indices[j]] += d2m*x[j];
    }
    
    // Store the angular flux
    for (unsigned int j=0; j<tensor_dim; ++j)
    {
#ifdef DEAL_II_USE_LARGE_INDEX_TYPE
      angular_flux[map->GID64(multivector_indices[j])] = x[j];
#else
      angular_flux[map->GID(multivector_indices[j])] = x[j];
#endif
    } 

    // Add the cells to the cells of the current task where the angular 
    // flux has been computed.
    cells_in_task.insert(cell);
  }

  // Send angular_flux to the waiting task
  scheduler->send_angular_flux(task,buffers,requests,angular_flux);

  // Delete all required_dofs map that is now useless
  task.clear_required_dofs();
}


template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::get_multivector_indices(
    std::vector<int> &dof_indices,
    active_cell_iterator const& cell) const
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
    for (unsigned int j=k+1; j<tensor_dim; ++j)
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
    for (unsigned int i=0; i<k; ++i)
      x[k] -= x[i]*A[k][i];
    x[k] /= A[k][k];
  }

  // Solve the linear equation Ux=y, where y is the solution
  // obtained above of x=b and U is an upper triangular matrix.
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
