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
        unsigned int max_iter,discrete_time tol);

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
    /// Enum to simplify access to the different elements of the tuple in
    /// schedule: (pointer to the task, start time of the task, end time of the
    /// task, rank of the task).
    enum {TASK,START,END,RANK};

    //TODO the initial scheduling may not be optimal. If the vectors where the
    //search are done are too big, the heuristic may be slow.
    /// Create the initial scheduling that will be used as a initial guess.
    void create_initial_scheduling();

    /// Send the end time of the task to the waiting tasks.
    void send_task_done(Task const &task,discrete_time task_end_time,
        std::list<discrete_time*> &buffers,std::list<MPI_Request*> &requests) const;

    /// Receive the end time of the task sent by send_tasks_done and add the 
    /// task to tasks_done and required_tasks_schedule.
    void receive_tasks_done(std::unordered_set<Task::global_id,
        boost::hash<Task::global_id>> &tasks_done, 
        std::vector<std::pair<Task::global_id,discrete_time>>
        &required_tasks_schedule);

    /// Perform the backward iteration, i.e., try to start every task as late as
    /// possible.
    void backward_iteration();

    /// Compute the rank used in the backward iteration.
    void compute_backward_ranks();

    /// This function returns true if the rank of first if higher than the rank
    /// of second. If the ranks are identical, the end time of the associated
    /// task is compared.
    bool decreasing_rank_comp(std::tuple<Task*,discrete_time,discrete_time,discrete_time> const &first,
        std::tuple<Task*,discrete_time,discrete_time,discrete_time> const &second);

    /// Sort the rank created by compute_backward_ranks in decreasing order.
    void decreasing_rank_sort();

    /// Schedule the tasks, using the sorted ranks, such that the tasks are
    /// started as last as possible.
    void backward_scheduling();

    /// Perform the forward iteration, i.e., try to start every task as early as
    /// possible.
    void forward_iteration();

    /// Compute the rank used in the backward iteration.
    void compute_forward_ranks();

    /// Sort the rank created by compute_forward_ranks in increasing order.
    void forward_sort();

    /// Schedule the tasks, using the sorted ranks, such that the tasks are
    /// started as early as possible.
    void forward_scheduling();

    /// Update schedule_id_map, which is a map between the local ID of the tasks
    /// and the positions in the schedule vector.
    void update_schedule_id_map();

    /// Build the required_tasks_schedule vector. The vector contains pairs of
    /// task global ID and end time of tasks that are required by the local
    /// tasks.
    void build_required_tasks_schedule(std::vector<std::pair<Task::global_id,
        discrete_time>> &required_tasks_schedule) const;

    /// Send the start time of the task to the required tasks during the
    /// backward scheduling. In backward scheduling, the sweep is done in
    /// reverse therefore required tasks and waiting tasks exchange role
    /// compared to the forward role.
    void send_start_time_required_tasks(Task const &task, discrete_time task_start_time,
        std::list<discrete_time*> &buffers,std::list<MPI_Request*> &requests) const;

    /// Receive the start time of the waiting tasks of the task during the
    /// backward scheduling. In backward scheduling, the sweep is done in
    /// reverse therefore required tasks and waiting tasks exchange role
    /// compared to the forward role.
    void receive_start_time_waiting_tasks(Task const &task,
        std::vector<std::pair<Task::global_id,discrete_time>> &waiting_tasks_schedule) const;

    /// Maximum number of CAP-PFB iterations.
    unsigned int max_iter;
    ///
    mutable unsigned int schedule_pos;
    /// Tolerance for the convergence of CAP-PFB.
    discrete_time tol;
    /// Global start time of the scheduling.
    discrete_time start_time;
    /// Global end time of the scheduling.
    discrete_time end_time;
    /// Best ordering of the tasks. This is the ordering that will be used
    /// during the sweeping.
    std::vector<Task*> best_schedule;
    /// Vector containing the ordered tasks, their start time, their end
    /// time, and their rank.
    // TODO this needs to be cleared later on
    std::vector<std::tuple<Task*,discrete_time,discrete_time,discrete_time>> schedule;
    // TODO This needs to be cleared later on
    /// Map between the local ID of the tasks and the positions in the schedule
    /// vector.
    std::unordered_map<unsigned int,unsigned int> schedule_id_map;

};

#endif

