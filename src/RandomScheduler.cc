/* Copyright (c) 2014, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "RandomScheduler.hh"

template <int dim,int tensor_dim>
RandomScheduler<dim,tensor_dim>::RandomScheduler(RTQuadrature const* quad,
    Epetra_MpiComm const* comm,ConditionalOStream const &pcout) :
  Scheduler<dim,tensor_dim>(quad,comm,pcout)
{
  this->pcout<<"Random Scheduler constructed."<<std::endl;
}


template <int dim,int tensor_dim>
void RandomScheduler<dim,tensor_dim>::start() const
{
  tasks_ready.clear();
  this->n_tasks_to_execute = this->tasks.size();
  for (unsigned int i=0; i<this->n_tasks_to_execute; ++i)
    tasks_to_execute.push_back(i);
}


template <int dim,int tensor_dim>
Task const* const RandomScheduler<dim,tensor_dim>::get_next_task() const
{
  // If tasks_ready is empty, we need to wait to receive data
  while (this->tasks_ready.size()==0)
  {
    this->receive_angular_flux();
    auto task_pos=tasks_to_execute.begin();
    while (task_pos!=tasks_to_execute.end())
    {
      if (this->tasks[*task_pos].is_ready()==true)
      {
        tasks_ready.push_back(*task_pos);
        task_pos = tasks_to_execute.erase(task_pos);
      }
      else
        ++task_pos;
    }
  }

  // Pop a task from the tasks_ready list and decrease the number of tasks
  // that are left to be executed by the current processor.
  --(this->n_tasks_to_execute);
  unsigned int i(tasks_ready.front());
  tasks_ready.pop_front();

  return &(this->tasks[i]);
}


//*****Explicit instantiations*****//
template class RandomScheduler<2,4>;
template class RandomScheduler<2,9>;
template class RandomScheduler<2,16>;
template class RandomScheduler<2,25>;
template class RandomScheduler<2,36>;
template class RandomScheduler<3,8>;
template class RandomScheduler<3,27>;
template class RandomScheduler<3,64>;
template class RandomScheduler<3,125>;
template class RandomScheduler<3,216>;
