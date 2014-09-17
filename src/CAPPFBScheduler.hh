/* Copyright (c) 2014, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _CAPPFBSCHEDULER_HH_
#define _CAPPFBSCHEDULER_HH_

#include <list>
#include <unordered_set>
#include <vector>
#include "mpi.h"
#include "boost/functional/hash/hash.hpp"
#include "Scheduler.hh"
#include "Task.hh"

// Forward declaration of Epetra_MpiComm and RTQuadrature since they are not use 
// by RandomScheduler directly.
class Epetra_MpiComm;
class RTQuadrature;

/**
 * This class is derived from Scheduler and implements the Cut Arc Preference
 * strategy (CAP) as vertex (task) ranking in the Parallel Forward-Backward
 * scheduler (PFB). CAP-PFB is described in <em>A new parallel algorithm for 
 * vertex priorities of data flow acyclic digraphs</em>, <b>Journal of
 * Supercomputing</b> (2014) 68:49-64, <em>Zeyao Mo, Aiqing Zhang, and Zhang
 * Yang</em>.
 */
template <int dim,int tensor_dim>
class CAPPFBScheduler : public Scheduler<dim,tensor_dim>
{
  public :
    /// Instead of using a floating point to represent the time, a discrete time is
    /// used where one unit of time represents one unit of work.
    typedef unsigned long long int discrete_time;
    typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;
    typedef std::pair<types::subdomain_id,unsigned int> global_id;

    /// Constructor. @p max_iter is the maximum number of iterations that the
    /// heuristic will perform, @p tol is the tolerance for convergence.
    CAPPFBScheduler(RTQuadrature const* quad,Epetra_MpiComm const* comm,
        unsigned int max_iter,unsigned int tol);

    /// Build patches of cells that will be sweep on, compute the sweep ordering
    /// on each of these patches, and finally build the tasks used in the sweep.
    void setup(const unsigned int n_levels,
        std::vector<FECell<dim,tensor_dim>> const* fecell_mesh_ptr,
        std::map<active_cell_iterator,unsigned int> const &cell_to_fecell_map)
      override;

    /// Get the scheduler ready to process tasks.
    void start() const override;

    /// Get a pointer to the next task which is ready.
    Task const* const get_next_task() const override;

  private :
    void create_initial_scheduling();

    void backward_iteration();

    void forward_iteration();

    void compute_backward_ranks(std::vector<discrete_time> &ranks);

    void decreasing_rank_sort(std::vector<discrete_time> &ranks);

    void backward_scheduling();

    void compute_forward_ranks(std::vector<discrete_time> &ranks);

    void forward_sort();

    void forward_scheduling();

    void update_schedule_id_map();

    void send_task_done(Task const &task,discrete_time task_start_time,
        discrete_time task_end_time,std::list<discrete_time*> &buffers,
        std::list<MPI_Request*> &requests) const;

    void receive_tasks_done(std::unordered_set<Task::global_id,
        boost::hash<Task::global_id>> &tasks_done, 
        std::vector<std::tuple<Task::global_id,discrete_time,discrete_time>>
        &required_tasks_schedule) const;

    void get_required_tasks_schedule(std::vector<std::pair<Task::global_id,
        discrete_time>> &required_tasks_schedule) const;

    void receive_start_time_waiting_tasks(Task const &task,
        std::vector<std::pair<Task::global_id,discrete_time>> &waiting_tasks_schedule) const;

    void send_start_time_required_tasks(Task const &task, discrete_time task_start_time,
        std::list<discrete_time*> &buffers,std::list<MPI_Request*> &requests) const;

    unsigned int max_iter;
    unsigned int tol;
    discrete_time start_time;
    discrete_time end_time;
    std::vector<std::tuple<Task*,discrete_time,discrete_time>> schedule;
    // This needs to be cleared later on
    std::unordered_map<unsigned int,unsigned int> schedule_id_map;

};

#endif

