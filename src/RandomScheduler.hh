/* Copyright (c) 2014, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _RANDOMSCHEDULER_HH_
#define _RANDOMSCHEDULER_HH_

#include <list>
#include "Scheduler.hh"
#include "Task.hh"

// Forward declaration of Epetra_MpiComm and RTQuadrature since they are not use 
// by RandomScheduler directly.
class Epetra_MpiComm;
class RTQuadrature;

/**
 * This class is derived from Scheduler and implements a scheduler where all the
 * tasks are executed in a random order.
 */
template <int dim,int tensor_dim>
class RandomScheduler : public Scheduler<dim,tensor_dim>
{
  public : 
    /// Constructor.
    RandomScheduler(RTQuadrature const* quad,Epetra_MpiComm const* comm,
        ConditionalOStream const &pcout);

    /// Get the scheduler ready to process tasks.
    void start() const override;

    /// Get a pointer to the next task which is ready.
    Task const* const get_next_task() const override;

  private :  
    /// List of tasks that are ready to be used by sweep. Because of the 
    /// Trilinos interface in Epetra_Operator, tasks_ready is made mutable 
    /// so it can be changed in a const function.
    mutable std::list<unsigned int> tasks_ready;
    /// List of tasks that are not in the task_ready list and that have not 
    /// been executed yet. Because of the Trilinos interface in Epetra_Operator, 
    /// tasks_ready is made mutable so it can be changed in a const function.
    mutable std::list<unsigned int> tasks_to_execute;
};

#endif
