/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _RADIATIVETRANSFER_HH_
#define _RADIATIVETRANSFER_HH_

#include <algorithm>
#include <cmath>
#include <list>
#include <set>
#include <utility>
#include <vector>
#include "mpi.h"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Operator.h"
#include "deal.II/base/exceptions.h"
#include "deal.II/base/quadrature_lib.h"
#include "deal.II/base/point.h"
#include "deal.II/distributed/tria.h"
#include "deal.II/dofs/dof_handler.h"
#include "deal.II/fe/fe_dgq.h"
#include "deal.II/fe/fe_values.h"
#include "deal.II/lac/full_matrix.h"
#include "deal.II/lac/trilinos_vector.h"
#include "deal.II/lac/vector.h"
#include "FECell.hh"
#include "Parameters.hh"
#include "RTQuadrature.hh"
#include "RTMaterialProperties.hh"
#include "Task.hh"

using namespace dealii;


/**
 *  This class derives from Epetra_Operator and implement the function Apply
 *  needed by the AztecOO solvers.
 */

template <int dim,int tensor_dim>
class RadiativeTransfer : public Epetra_Operator
{
  public :
    RadiativeTransfer(FE_DGQ<dim>* fe,
        parallel::distributed::Triangulation<dim>* triangulation,
        DoFHandler<dim>* dof_handler,Parameters* parameters,RTQuadrature* quad,
        RTMaterialProperties* material_properties,Epetra_MpiComm const* comm,
        Epetra_Map const* map);

    ~RadiativeTransfer();

    /// Create all the matrices that are need to solve the transport equation
    /// and compute the sweep ordering.
    void setup();

    void build_waiting_tasks_map();
    void build_local_waiting_tasks_map(Task &task,
        types::global_dof_index* recv_dof_buffer,int* recv_dof_disps_x,
        const unsigned int recv_dof_buffer_size);
    void build_required_tasks_map();
    void build_local_required_tasks_map(Task &task,
        types::global_dof_index* recv_dof_buffer,int* recv_dof_disps_x,
        const unsigned int recv_n_dofs_buffer);
    void get_task_local_dof_indices(Task &task,std::vector<types::global_dof_index> 
        &local_dof_indices);
    Task const* const get_next_task() const;
    void initialize_scheduler() const;
    void clear_scheduler() const;
    unsigned int get_n_tasks_to_execute() const;
    void free_buffers(std::list<double*> &buffers,std::list<MPI_Request*> &requests) 
      const;

    /// Return the result of the transport operator applied to x in. Return 0
    /// if successful.
    int Apply(Epetra_MultiVector const &x,Epetra_MultiVector &y) const;

    /// Compute the scattering given a flux.
    void compute_scattering_source(Epetra_MultiVector const &x) const;

    /// Perform a sweep. If group_flux is nullptr is false, the surfacic and 
    /// the volumetric sources are not included in the sweep.
    /// @todo The sweep can be optimized to use less memory (only keep the
    /// front wave).
    void sweep(Task const &task,std::list<double*> &buffers,
        std::list<MPI_Request*> &requests,Epetra_MultiVector &flux_moments,
        Epetra_MultiVector &psi,
        std::vector<TrilinosWrappers::MPI::Vector> const* const group_flux=nullptr) 
      const;

    /// This method is not implemented.
    int SetUseTranspose(bool UseTranspose) {return 0;};

    /// This method is not implemented.
    int ApplyInverse(Epetra_MultiVector const &x,Epetra_MultiVector &y) const
    {return 0;};

    /// This method is not implemented.
    double NormInf() const {return 0.;};

    /// Return a character string describing the operator.
    char const* Label() const {return "radiative_transfer";};

    /// Return the UseTranspose setting (always false).
    bool UseTranspose() const {return "false";};

    /// Retun true if this object can provide an approximate Inf-norm (always
    /// false).
    bool HasNormInf() const {return false;};

    /// Return pointer to the Epetra_Comm communicator associated with this
    /// operator.
    Epetra_Comm const& Comm() const;

    /// Return the Epetra_Map object associated with the domain of this
    /// operator.
    Epetra_Map const& OperatorDomainMap() const;

    /// Return the Epetra_Map object associated with the range of this
    /// operator.
    Epetra_Map const& OperatorRangeMap() const;
    
    /// Set the current group.
    void set_group(unsigned int g);    

  private :
    void build_global_required_tasks();
    void receive_angular_flux() const;
    void send_angular_flux(Task const &task,std::list<double*> &buffers,
        std::list<MPI_Request*> &requests,
        std::unordered_map<types::global_dof_index,double> &angular_flux) const;

    /// Compute the ordering of the cell for the sweeps.
    void compute_sweep_ordering();
    
    /// Compute the scattering source due to the upscattering and the
    /// downscattering to the current group.
    void compute_outer_scattering_source(Tensor<1,tensor_dim> &b,
        std::vector<TrilinosWrappers::MPI::Vector> const* const group_flux,
        FECell<dim,tensor_dim> const* const fecell,const unsigned int idir) const;

    /// Get the local indices (used by the Epetra_MultiVector) of the dof 
    /// associated to a given cell.
    void get_multivector_indices(std::vector<int> &dof_indices,
    typename DoFHandler<dim>::active_cell_iterator const& cell) const;

    /**
     * This routine uses Crout's method with pivoting to decompose the matrix
     * \f$A\f$ into lower triangulat matrix \f$L\f$ and an unit upper
     * triangular matrix \f$U\f$ such that $\f$A=LU\f$. The matrices \f$L\f$
     * such replace the matrix \f$A\f$ so that the original matrix \f$A\f$ is
     * destroyed. In Crout's method the diagonal element of \f$U\f$ are ones
     * and are not stored.
     */
    void LU_decomposition(Tensor<2,tensor_dim> &A,
        Tensor<1,tensor_dim,unsigned int> &pivot) const;

    /// This routive is called after the matrix \f$A\f$ has been decomposed
    /// into \f$L\f$ and \f$U\f$.
    void LU_solve(Tensor<2,tensor_dim> const &A,Tensor<1,tensor_dim> &b,
        Tensor<1,tensor_dim> &x,Tensor<1,tensor_dim,unsigned int> const &pivot) const;

    // Pointers because of Trilinos
    mutable unsigned int n_tasks_to_execute;
    mutable std::list<unsigned int> tasks_ready;

    /// Number of moments.
    unsigned int n_mom;
    /// Current group.
    unsigned int group;
    /// Number of groups.
    unsigned int n_groups;
    /// Epetra communicator.
    Epetra_MpiComm const* comm;
    /// Pointer to Epetra_Map associated to flux_moments and group_flux
    Epetra_Map const* map;

    mutable std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
      std::unordered_map<types::global_dof_index,unsigned int>,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>> global_required_tasks;
    /// Scattering source for each moment.
    std::vector<Vector<double>*> scattering_source;
    /// FECells owned by the current processor.
    std::vector<FECell<dim,tensor_dim>> fecell_mesh;
    /// Tasks owned by the current processor.
    std::vector<Task> tasks;
    /// Pointer to the discontinuous finite element object.
    FE_DGQ<dim>* fe;
    /// Pointer to the distributed triangulation.
    parallel::distributed::Triangulation<dim>* triangulation;
    /// Pointer to the dof_handler.
    DoFHandler<dim>* dof_handler;
    /// Pointer to the parameters.
    Parameters* parameters;
    /// Pointer to the quadrature.
    RTQuadrature* quad;
    /// Pointer to the material property.
    RTMaterialProperties* material_properties;
};

template <int dim,int tensor_dim>
inline Epetra_Comm const& RadiativeTransfer<dim,tensor_dim>::Comm() const
{
  return *comm;
}

template <int dim,int tensor_dim>
inline Epetra_Map const& RadiativeTransfer<dim,tensor_dim>::OperatorDomainMap() const
{
  return *map;
}

template <int dim,int tensor_dim>
inline Epetra_Map const& RadiativeTransfer<dim,tensor_dim>::OperatorRangeMap() const
{
  return *map;
}

template <int dim,int tensor_dim>
inline void RadiativeTransfer<dim,tensor_dim>::set_group(unsigned int g)
{
  group = g;
}

template <int dim,int tensor_dim>
inline unsigned int RadiativeTransfer<dim,tensor_dim>::get_n_tasks_to_execute() const
{
  return n_tasks_to_execute;
}

#endif
