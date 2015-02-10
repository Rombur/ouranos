/* Copyright (c) 2014, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _SCHEDULER_HH_
#define _SCHEDULER_HH_

#include <list>
#include <map>
#include <unordered_set>
#include <utility>
#include "boost/functional/hash/hash.hpp"
#include "Epetra_Comm.h"
#include "deal.II/base/conditional_ostream.h"
#include "deal.II/base/exceptions.h"
#include "deal.II/dofs/dof_handler.h"
#include "FECell.hh"
#include "RTQuadrature.hh"
#include "Task.hh"

using namespace dealii;


/**
 * This is the base class for all the schedulers. This class takes care of the
 * scheduling of the tasks and of the communication between tasks.
 */ 
template <int dim,int tensor_dim>
class Scheduler
{
  public :
    typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;

    /// Constructor. 
    Scheduler(RTQuadrature const* quad,Epetra_MpiComm const* comm,
        ConditionalOStream const &pcout);

    /// Destructor.
    virtual ~Scheduler() {};

    /// Build patches of cells that will be sweep on, compute the sweep ordering
    /// on each of these patches, and finally build the tasks used in the sweep.
    virtual void setup(const unsigned int n_levels,
        std::vector<FECell<dim,tensor_dim>> const* fecell_mesh_ptr,
        std::map<active_cell_iterator,unsigned int> const &cell_to_fecell_map);
    
    /// Get the scheduler ready to process tasks.
    virtual void start() const = 0;

    /// Send the angular flux computed in task to all the waiting tasks.
    void send_angular_flux(Task const &task,std::list<double*> &buffers,
        std::list<MPI_Request*> &requests,
        std::unordered_map<types::global_dof_index,double> & angular_flux) const;

    /// Free the buffers and MPI_Request used to send MPI messages.
    template <typename data_type>
    void free_buffers(std::list<data_type*> &buffers,std::list<MPI_Request*> &requests) 
      const;

    /// Return the number of tasks left to execute.
    unsigned int get_n_tasks_to_execute() const;

    /// Get a pointer to the next task which is ready.
    virtual Task const* const get_next_task() const = 0;

  protected :
    /// Received the angular flux from a required task.
    void receive_angular_flux() const;

    /// Epetra communicator.
    Epetra_MpiComm const* comm;
    /// MPI communicator.
    MPI_Comm mpi_comm;
    /// Conditional output stream. This allows to have only one processor do
    /// the output.
    ConditionalOStream pcout;
    /// Number of tasks left to execute. Because of the Trilinos interface
    /// in Epetra_Operator, n_tasks_to_execute is made mutable so it
    /// can be changed in a const function.
    mutable unsigned int n_tasks_to_execute;
    /// Tasks owned by the current processor.
    std::vector<Task> tasks;
    /// The key of this map is the subdomain_id and the task id of the required
    /// task, which is on another processor, and the value is a vector of the 
    /// position of the waiting tasks in the local vector of tasks. Because of 
    /// the Trilinos interface in Epetra_Operator, ghost_required_tasks is made 
    /// mutable so it can be can be changed in a const function.
    mutable std::unordered_map<Task::global_id,std::vector<unsigned int>,
            boost::hash<Task::global_id>> ghost_required_tasks;
    /// This vector is used to store the position in a received MPI message from
    /// a given task of a given dof. Because of the Trilinos interface in
    /// Epetra_Operator, global_required_tasks is made mutable so it can be
    /// can be changed in a const function.
    mutable std::vector<std::tuple<types::subdomain_id,unsigned int,
            std::unordered_map<types::global_dof_index,unsigned int>>> global_required_tasks;

  private :
    /// Get all the dof indices associated to the given task.
    std::unordered_set<types::global_dof_index> get_task_local_dof_indices(Task &task);

    /// Build the required_tasks map associated to the given task.
    void build_local_required_tasks_map(Task &task,
        types::global_dof_index* recv_dof_buffer,int* recv_dof_disps_x,
        const unsigned int recv_n_dofs_buffer);

    /// Build the waiting_tasks map associated to the task.
    void build_local_waiting_tasks_map(Task &task,
        types::global_dof_index* recv_dof_buffer,int* recv_dof_disps_x,
        const unsigned int recv_dof_buffer_size);

    /// Build the global_required_tasks map.
    void build_global_required_tasks();

    /// Build local_tasks_map.
    void build_local_tasks_map();

    /// Build the required_tasks maps for all the tasks owned by a processor.
    void build_required_tasks_maps();

    /// Build the waiting_tasks maps for all the tasks owned by a processor.
    void build_waiting_tasks_maps();                                       

    /// Build convex patches of cells by going up the tree of cells. All the
    /// cells in a patch are on the same processors. The coarsest patches
    /// corresponds to the cells of the coarse mesh.
    void build_cell_patches(const unsigned int n_levels,
        std::map<active_cell_iterator,unsigned int> const &cell_to_fecell_map,
        std::list<std::list<unsigned int>> &cell_patches) const;

    /// Recursive function that goes down the tree of descendants of the current
    /// cell. The function returns false if one of the descendants is not
    /// locally owned. It also adds the active descendants to a patch.
    bool explore_descendants_tree(typename DoFHandler<dim>::cell_iterator const &current_cell,
        std::list<active_cell_iterator> &active_descendants) const;

    /// Compute the ordering of the cells on each patch for the sweeps and
    /// create the tasks.
    void compute_sweep_ordering(
        std::map<active_cell_iterator,unsigned int> const &cell_to_fecell_map,
        std::list<std::list<unsigned int>> &cell_patches);

    /// Pointer to the quadrature.
    RTQuadrature const* quad;
    /// If the waiting task is on the current processor, this map can be used
    /// to find the position of the task in the tasks vector. The key of the map
    /// is the task id of the waiting task and the value is the position in
    /// the local tasks vector. Because of the Trilinos interface in 
    /// Epetra_Operator, local_tasks_map is made mutable so it can be can be 
    /// changed in a const function.
    mutable std::unordered_map<unsigned int,unsigned int> local_tasks_map;
    /// Pointer to the FECells owned by the current processor.
    std::vector<FECell<dim,tensor_dim>> const* fecell_mesh;
};

template <int dim,int tensor_dim>
inline unsigned int Scheduler<dim,tensor_dim>::get_n_tasks_to_execute() const
{
  return n_tasks_to_execute;
}


template <int dim,int tensor_dim>
template <typename data_type>
void Scheduler<dim,tensor_dim>::free_buffers(
    std::list<data_type*> &buffers,std::list<MPI_Request*> &requests) const
{  
  typename std::list<data_type*>::iterator buffers_it(buffers.begin());
  typename std::list<data_type*>::iterator buffers_end(buffers.end());
  std::list<MPI_Request*>::iterator requests_it(requests.begin());
  while (buffers_it!=buffers_end)
  {  
    // If the message has been received, the buffer and the request are delete. 
    // Otherwise, we just try the next buffer.
    int flag;
    MPI_Test(*requests_it,&flag,MPI_STATUS_IGNORE);
    if (flag==true)
    {
      delete [] *buffers_it;
      delete *requests_it;
      buffers_it = buffers.erase(buffers_it);
      requests_it = requests.erase(requests_it);
    }                            
    else
    {                               
      ++buffers_it;
      ++requests_it;
    }
  }
}      

#endif

