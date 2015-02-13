/* Copyright (c) 2014-2015, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "CAPPFBScheduler.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>

template <int dim,int tensor_dim>
CAPPFBScheduler<dim,tensor_dim>::CAPPFBScheduler(RTQuadrature const* quad,
    Epetra_MpiComm const* comm,ConditionalOStream const &pcout,
    unsigned int max_iter) :
  Scheduler<dim,tensor_dim>(quad,comm,pcout),
  max_iter(max_iter),
  start_time(0),
  end_time(0)
{
  this->pcout<<"CAP-PFB Scheduler constructed."<<std::endl;
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::setup(const unsigned int n_levels,
    std::vector<FECell<dim,tensor_dim>> const* fecell_mesh_ptr,
    std::map<active_cell_iterator,unsigned int> const &cell_to_fecell_map)
{
  Scheduler<dim,tensor_dim>::setup(n_levels,fecell_mesh_ptr,cell_to_fecell_map);

  // Create the initial scheduling used as guess.
  create_initial_scheduling();
  this->pcout<<"Initial sweep: Time need to sweep through the mesh: "
    <<end_time-start_time<<std::endl;

  // Iterate for a given number of iterations.
  const unsigned int n_tasks(schedule.size());
  best_schedule.resize(n_tasks,NULL);
  copy_best_schedule(true);
  discrete_time best_schedule_time(end_time-start_time);
  for (unsigned int i=0; i<max_iter; ++i)
  {
    backward_iteration();
    this->pcout<<"Backward sweep: Time need to sweep through the mesh: "
      <<end_time-start_time<<std::endl;
    if (best_schedule_time>(end_time-start_time))
    {
      best_schedule_time = end_time-start_time;
      copy_best_schedule(false);
    }
    forward_iteration();
    this->pcout<<"Forward sweep: Time need to sweep through the mesh: "
      <<end_time-start_time<<std::endl;
    if (best_schedule_time>(end_time-start_time))
    {
      best_schedule_time = end_time-start_time;
      copy_best_schedule(true);
    }
  }

  this->pcout<<"Best sweep: Time need to sweep through the mesh: "
    <<end_time-start_time<<std::endl;

  // Clear schedule, schedule_id_map, and waiting_tasks in each tasks.
  clear();
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::start() const
{
  this->n_tasks_to_execute = this->tasks.size();
  next_task_pos = 0U;
}


template <int dim,int tensor_dim>
Task const* const CAPPFBScheduler<dim,tensor_dim>::get_next_task() const
{
  while (best_schedule[next_task_pos]->is_ready()==false)
    this->receive_angular_flux();

  // Decrease the number of tasks that are left to be executed by the current
  // processor.
  --(this->n_tasks_to_execute);

  ++next_task_pos;

  return best_schedule[next_task_pos-1];
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::create_initial_scheduling()
{
  this->n_tasks_to_execute = this->tasks.size();
  std::list<discrete_time*> buffers;
  std::list<MPI_Request*> requests;
  unsigned int n_tasks_done(0);
  std::unordered_set<Task::global_id,boost::hash<Task::global_id>> tasks_done;
  std::vector<std::vector<discrete_time>> required_tasks_end_time(this->n_tasks_to_execute);
  // Loop as long as there are tasks to execute.
  while (n_tasks_done<this->n_tasks_to_execute)
  {
    for (auto & task : this->tasks)
    {
      // Check that the task has not been executed already.
      if (tasks_done.count(task.get_global_id())==0)
      {
        // Check that the task can be executed, i.e., all the required tasks are
        // done.
        bool ready(true);
        for (auto required_task=task.get_required_tasks_cbegin();
            required_task<task.get_required_tasks_cend(); ++required_task)
        {
          const Task::global_id required_task_id(std::get<0>(*required_task),
              std::get<1>(*required_task));
          if (tasks_done.count(required_task_id)==0)
          {
            ready = false;
            break;
          }
        }
        // If the task can be executed, search for the time when the latest
        // required task is finished.
        if (ready==true)
        {
          discrete_time task_start_time(0);
          // Search in the tasks that are not owned by the processor.
          for (auto const & task_end_time : required_tasks_end_time[task.get_local_id()])
            if (task_end_time>task_start_time)
              task_start_time = task_end_time;

          // Take the maximum of task_start_time and end_time to be sure that
          // the processor is idle long enough.
          task_start_time = std::max(task_start_time,end_time);
          // The number of work units is the number of cells in the patch.
          end_time = task_start_time + task.get_sweep_order_size();
          schedule.push_back(std::tuple<Task*,discrete_time,discrete_time,
              discrete_time>(&task,task_start_time,end_time,0));
          schedule_id_map[task.get_local_id()] = n_tasks_done;
          tasks_done.insert(task.get_global_id());
          ++n_tasks_done;
          // Tell the other processors that the task has been done, i.e.
          // add it to the tasks_done on other processors and update
          // required_tasks_end_time.
          send_task_done(task,end_time,buffers,requests);
        }
      }
    }
    // Check if some tasks required have been executed by others processors
    // and free usunused buffers.
    receive_tasks_done(tasks_done,required_tasks_end_time);
    this->free_buffers(buffers,requests);
  }
  // Free all the buffers left.
  while (buffers.size()!=0)
    this->free_buffers(buffers,requests);

  // Update start time and end time.
  MPI_Allreduce(MPI_IN_PLACE,&start_time,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,this->mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE,&end_time,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,this->mpi_comm);
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::copy_best_schedule(bool forward)
{
  const unsigned int n_tasks(best_schedule.size());
  if (forward==true)
    for (unsigned int i=0; i<n_tasks; ++i)
      best_schedule[i] = std::get<TASK>(schedule[i]);
  else
    for (unsigned int i=0; i<n_tasks; ++i)
      best_schedule[i] = std::get<TASK>(schedule[n_tasks-1-i]);
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::send_task_done(Task const &task,
    discrete_time task_end_time,std::list<discrete_time*> &buffers,
    std::list<MPI_Request*> &requests) const
{
  const types::subdomain_id source(task.get_subdomain_id());
  const unsigned int tag(task.get_local_id());

  // Loop over all the waiting processors
  std::vector<Task::subdomain_dof_pair>::const_iterator map_it(
      task.get_waiting_subdomains_cbegin());
  std::vector<Task::subdomain_dof_pair>::const_iterator map_end(
      task.get_waiting_subdomains_cend());
  for (; map_it!=map_end; ++map_it)
  {
    types::subdomain_id destination(map_it->first);
    // If source and destination processors are different, MPI is used
    if (source!=destination)
    {
      //TODO message too short :/
      int count(1);
      discrete_time* buffer = new discrete_time [count];
      buffer[0] = task_end_time;
      buffers.push_back(buffer);
      MPI_Request* request = new MPI_Request;
      requests.push_back(request);

      MPI_Isend(buffer,count,MPI_UNSIGNED_LONG_LONG,destination,tag,this->mpi_comm,request);
    }
  }
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::receive_tasks_done(
    std::unordered_set<Task::global_id,boost::hash<Task::global_id>> &tasks_done,
    std::vector<std::vector<discrete_time>> &required_tasks_end_time)
{
  // Loop on the global_required_tasks map
  std::vector<std::tuple<types::subdomain_id,unsigned int,
    std::unordered_map<types::global_dof_index,unsigned int>>>::iterator
      global_map_it(this->global_required_tasks.begin());
  std::vector<std::tuple<types::subdomain_id,unsigned int,
    std::unordered_map<types::global_dof_index,unsigned int>>>::iterator
      global_map_end(this->global_required_tasks.end());
  for (; global_map_it!=global_map_end; ++global_map_it)
  {
    types::subdomain_id source(std::get<TASK>(*global_map_it));
    unsigned int tag(std::get<1>(*global_map_it));
    int flag;

    // Check if we can receive a message
    MPI_Iprobe(source,tag,this->mpi_comm,&flag,MPI_STATUS_IGNORE);

    if (flag==true)
    {
      const Task::global_id ghost_task(source,tag);
      //TODO message to short
      int count(1);
      discrete_time* buffer = new discrete_time [count];

      // Receive the message
      MPI_Recv(buffer,count,MPI_UNSIGNED_LONG_LONG,source,tag,this->mpi_comm,MPI_STATUS_IGNORE);

      tasks_done.insert(ghost_task);
      // Set the time by looping on the tasks that are waiting for
      // the ghost cell
      std::vector<unsigned int>::const_iterator required_tasks_it(
          this->ghost_required_tasks[ghost_task].cbegin());
      std::vector<unsigned int>::const_iterator required_tasks_end(
          this->ghost_required_tasks[ghost_task].cend());
      for (; required_tasks_it!=required_tasks_end; ++required_tasks_it)
        required_tasks_end_time[*required_tasks_it].push_back(buffer[0]);

      delete [] buffer;
    }
  }
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::backward_iteration()
{
  compute_backward_ranks();
  decreasing_rank_sort();
  backward_scheduling();
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::compute_backward_ranks() 
{
  // We need the updated end times of the tasks required to execute the tasks
  // locally owned.
  std::vector<std::pair<Task::global_id,discrete_time>> required_tasks_schedule;
  build_required_tasks_schedule(required_tasks_schedule);
  const types::subdomain_id current_subdomain(std::get<TASK>(schedule[0])->get_subdomain_id());

  for (auto const & scheduled_task : schedule)
  {
    const unsigned int current_task_pos(schedule_id_map[std::get<TASK>(scheduled_task)->get_local_id()]);
    std::get<RANK>(schedule[current_task_pos]) = 0;
    for (auto required_task=std::get<TASK>(scheduled_task)->get_required_tasks_cbegin();
        required_task<std::get<TASK>(scheduled_task)->get_required_tasks_cend(); ++required_task)
    {
      // Search for the time when the latest required task is finished, first
      // among the tasks locally owned and after among the tasks owned by other
      // processors.
      Task::global_id required_task_id(std::get<0>(*required_task),std::get<1>(*required_task));
      discrete_time candidate_time(0);
      if (required_task_id.first==current_subdomain)
      {
        const unsigned int pos(schedule_id_map[required_task_id.second]);
        candidate_time = std::get<RANK>(schedule[pos]);
      }
      else
      {
        // Search among the tasks that are not owned by the processor.
        for (auto const & scheduled_required_task : required_tasks_schedule)
        {
          const Task::global_id task_id(scheduled_required_task.first);
          if (task_id==required_task_id)
          {
            candidate_time = scheduled_required_task.second;
            break;
          }
        }
      }

      if (candidate_time>std::get<RANK>(schedule[current_task_pos]))
        std::get<RANK>(schedule[current_task_pos]) = candidate_time;
    }
  }
}


template <int dim,int tensor_dim>
bool CAPPFBScheduler<dim,tensor_dim>::decreasing_rank_comp(
    std::tuple<Task*,discrete_time,discrete_time,discrete_time> const &first,
    std::tuple<Task*,discrete_time,discrete_time,discrete_time> const &second) const
{
  // Sort the tasks by decreasing rank. The ties are break by looking at the end
  // time of the tasks.
  if (std::get<RANK>(first)!=std::get<RANK>(second))
    return (std::get<RANK>(first)>std::get<RANK>(second));
  else
    return (std::get<END>(first)>std::get<END>(second));
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::decreasing_rank_sort()
{
  // Sort the tasks by decreading rank
  std::sort(schedule.begin(),schedule.end(),
      std::bind(&CAPPFBScheduler<dim,tensor_dim>::decreasing_rank_comp,this,
        std::placeholders::_1,std::placeholders::_2));

  // Need to update schedule_id_map since schedule has been reordered
  update_schedule_id_map();
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::backward_scheduling()
{
  std::list<discrete_time*> buffers;
  std::list<MPI_Request*> requests;

  // Create a vector that represents the discrete time when the processor is
  // idle. We don't want the vector to be too large (small start_time) but we
  // need to allow the new scheduling to take more time than the current one.
  int n_procs(0);
  MPI_Comm_size(this->mpi_comm,&n_procs);
  end_time += std::sqrt(10*n_procs);
  discrete_time candidate_start_time(end_time);
  std::vector<bool> idle_processor(end_time-start_time,true);

  // Tasks that are waiting for the local tasks but are non-local.
  std::vector<std::pair<Task::global_id,discrete_time>> waiting_tasks_schedule;

  for (auto & schedule_elem : schedule) 
  {
    discrete_time candidate_time(end_time);
    Task* task(std::get<TASK>(schedule_elem));
    unsigned int n_waiting_tasks(task->get_n_waiting_tasks());
    // Check if the waiting tasks are done. Since this sweep is done backward
    // the waiting tasks needs to be "done" before the task can be executed.
    if (n_waiting_tasks!=0)
    {
      bool ready(false);
#ifdef DEBUG
      double while_start_time(MPI_Wtick());
#endif
      while (ready==false)
      {
#ifdef DEBUG
        double while_current_time(MPI_Wtick());
        int rank(0);
        MPI_Comm_rank(this->mpi_comm,&rank);
        std::string error_message("No improvement in Backward Scheduling during ");
        error_message += "the last 5 minutes on processor: ";
        error_message += std::to_string(rank);
        Assert((while_current_time-while_start_time)<600.,
            ExcMessage(error_message));
#endif
        ready = true;
        
        // Receive start time of waiting tasks
        receive_start_time_waiting_tasks(*task,waiting_tasks_schedule);

        for (unsigned int j=0; (j<n_waiting_tasks) && (ready==true); ++j)
        {
          Task::global_id waiting_task_id(task->get_waiting_tasks_global_id(j));
          // If the waiting task is local, search in schedule
          if (std::get<0>(waiting_task_id)==task->get_subdomain_id())
          {
            for (auto const &scheduled_task : schedule)
            {
              if (waiting_task_id==std::get<TASK>(scheduled_task)->get_global_id())
              {
                if (candidate_time>std::get<START>(scheduled_task))
                  candidate_time = std::get<START>(scheduled_task);
                
                break;
              }
            }
          }
          // If the waiting task is not local, search in waiting_tasks_schedule
          else
          {
            bool found(false);
            for (auto const & scheduled_waiting_task : waiting_tasks_schedule)
            { 
              if (waiting_task_id==scheduled_waiting_task.first)
              {
                if (candidate_time>scheduled_waiting_task.second)
                  candidate_time = scheduled_waiting_task.second;
                found = true;

                break;
              }
            }
            // If the task was not found, ready is set to false.
            ready = found;
          }
        }
      }
    }

    // Look for a time when we have ressource to do the computation.
    bool enough_free_time(false);
    // The execution time corresponds to the number of cells in the local sweep 
    // order.
    const unsigned int execution_time(task->get_sweep_order_size());
    while (enough_free_time==false)
    {
      enough_free_time = true;
      for (unsigned int i=0; i<execution_time; ++i)
      {
        if (idle_processor[candidate_time-i-start_time]==false)
        {
          candidate_time -= i+1;
          enough_free_time = false;

          break;
        }
      }
    }
    // Mark the processor as busy.
    for (unsigned int i=0; i<execution_time; ++i)
      idle_processor[candidate_time-i-start_time] = false;

    // Update the start and end time of the task.
    std::get<END>(schedule_elem) = candidate_time;
    std::get<START>(schedule_elem) = candidate_time - execution_time;
    if (std::get<START>(schedule_elem)<candidate_start_time)
      candidate_start_time = std::get<START>(schedule_elem);

    // Send the start time of the tasks to all the required tasks.
    send_start_time_required_tasks(*task,std::get<START>(schedule_elem),buffers,
        requests);
    this->free_buffers(buffers,requests);
  }

  // Wait for all the buffers to be free before exiting the function.
  while (buffers.size()!=0)
    this->free_buffers(buffers,requests);

  // Update start time.
  start_time = candidate_start_time;
  MPI_Allreduce(MPI_IN_PLACE,&start_time,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,this->mpi_comm);
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::forward_iteration()
{
  compute_forward_ranks();
  increasing_rank_sort();
  forward_scheduling();
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::compute_forward_ranks() 
{
  // We need the updated start times of the tasks waiting for the tasks locally 
  // owned to be executed.
  std::vector<std::pair<Task::global_id,discrete_time>> waiting_tasks_schedule;
  build_waiting_tasks_schedule(waiting_tasks_schedule);
  const types::subdomain_id current_subdomain(std::get<TASK>(schedule[0])->get_subdomain_id());

  for (auto const & scheduled_task : schedule)
  {
    const unsigned int current_task_pos(schedule_id_map[std::get<TASK>(scheduled_task)->get_local_id()]);
    std::get<RANK>(schedule[current_task_pos]) = -1;
    for (auto waiting_task=std::get<TASK>(scheduled_task)->get_waiting_tasks_cbegin();
        waiting_task<std::get<TASK>(scheduled_task)->get_waiting_tasks_cend(); ++waiting_task)
    {
      // Search for the time when the earliest waiting task is started, first
      // among the tasks locally owned and after among the tasks owned by other
      // processors.
      Task::global_id waiting_task_id(std::get<0>(*waiting_task),std::get<1>(*waiting_task));
      discrete_time candidate_time(0);
      if (waiting_task_id.first==current_subdomain)
      {
        unsigned int pos(schedule_id_map[waiting_task_id.second]);
        candidate_time = std::get<RANK>(schedule[pos]);
      }
      else
      {
        // Search among the tasks that are not owned by the processor.
        for (auto const & scheduled_waiting_task : waiting_tasks_schedule)
        {
          const Task::global_id task_id(scheduled_waiting_task.first);
          if (task_id==waiting_task_id)
          {
            candidate_time = scheduled_waiting_task.second;
            break;
          }
        }
      }

      if (candidate_time<std::get<RANK>(schedule[current_task_pos]))
        std::get<RANK>(schedule[current_task_pos]) = candidate_time;
    }
  }
}


template <int dim,int tensor_dim>
bool CAPPFBScheduler<dim,tensor_dim>::increasing_rank_comp(
    std::tuple<Task*,discrete_time,discrete_time,discrete_time> const &first,
    std::tuple<Task*,discrete_time,discrete_time,discrete_time> const &second) const
{
  // Sort the tasks by increasing rank. The ties are break by looking at the
  // start time of the tasks.
  if (std::get<RANK>(first)!=std::get<RANK>(second))
    return (std::get<RANK>(first)<std::get<RANK>(second));
  else
    return (std::get<START>(first)<std::get<START>(second));
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::increasing_rank_sort()
{
  // Sort the tasks by decreading rank
  std::sort(schedule.begin(),schedule.end(),
      std::bind(&CAPPFBScheduler<dim,tensor_dim>::increasing_rank_comp,this,
        std::placeholders::_1,std::placeholders::_2));

  // Need to update schedule_id_map since schedule has been reordered
  update_schedule_id_map();
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::forward_scheduling()
{
  std::list<discrete_time*> buffers;
  std::list<MPI_Request*> requests;

  // Create a vector that represents the discrete time when the processor is
  // idle. We don't want the vector to be too large (small start_time) but we
  // need to allow the new scheduling to take more time than the current one.
  int n_procs(0);
  MPI_Comm_size(this->mpi_comm,&n_procs);
  end_time += std::sqrt(10*n_procs);
  discrete_time candidate_end_time(start_time);
  std::vector<bool> idle_processor(end_time-start_time,true);

  // Tasks that are required for the local tasks but are non-local.
  std::vector<std::pair<Task::global_id,discrete_time>> required_tasks_schedule;

  for (auto & schedule_elem : schedule)
  {
    discrete_time candidate_time(start_time);
    Task* task(std::get<TASK>(schedule_elem));
    unsigned int n_required_tasks(task->get_n_required_tasks());
    // Check if the required tasks are done.
    if (n_required_tasks!=0)
    {
      bool ready(false);
#ifdef DEBUG
      double while_start_time(MPI_Wtick());
#endif
      while (ready==false)
      {
#ifdef DEBUG
        double while_current_time(MPI_Wtick());
        int rank(0);
        MPI_Comm_rank(this->mpi_comm,&rank);
        std::string error_message("No improvement in Forward Scheduling during ");
        error_message += "the last 5 minutes on processor: ";
        error_message += std::to_string(rank);
        Assert((while_current_time-while_start_time)<600.,
            ExcMessage(error_message));
#endif
        ready = true;
        // Receive end time of required tasks
        receive_end_time_required_tasks(*task,required_tasks_schedule);

        for (unsigned int j=0; (j<n_required_tasks) && (ready==true); ++j)
        {
          Task::global_id required_task_id(task->get_required_tasks_global_id(j));
          // If the waiting task is local, search in schedule
          if (std::get<0>(required_task_id)==task->get_subdomain_id())
          {
            for (auto const &scheduled_task : schedule)
            {
              if (required_task_id==std::get<TASK>(scheduled_task)->get_global_id())
              {
                if (candidate_time<std::get<END>(scheduled_task))
                  candidate_time = std::get<END>(scheduled_task);

                break;
              }
            }
          }
          // If the waiting task is not local, search in required_tasks_schedule
          else
          {
            bool found(false);
            for (auto const & scheduled_required_task : required_tasks_schedule)
            {
              if (required_task_id==scheduled_required_task.first)
              {
                if (candidate_time<scheduled_required_task.second)
                  candidate_time = scheduled_required_task.second;
                found = true;

                break;
              }
            }
            // If the task was not found, ready is set to false.
            ready = found;
          }
        }
      }
    }

    // Look for a time when we have ressource to do the computation.
    bool enough_free_time(false);
    // The execution time corresponds to the number of cells in the local sweep
    // order.
    const unsigned int execution_time(task->get_sweep_order_size());
    while (enough_free_time==false)
    {
      enough_free_time = true;
      for (unsigned int i=0; i<execution_time; ++i)
      {
        if (idle_processor[candidate_time+i-start_time]==false)
        {
          candidate_time += i+1;
          enough_free_time = false;

          break;
        }
      }
    }
    // Mark the processor as busy.
    for (unsigned int i=0; i<execution_time; ++i)
      idle_processor[candidate_time+i-start_time] = false;

    // Update the start and end time of the task.
    std::get<START>(schedule_elem) = candidate_time;
    std::get<END>(schedule_elem) = candidate_time + execution_time;
    if (std::get<END>(schedule_elem)>candidate_end_time)
      candidate_end_time = std::get<END>(schedule_elem);

    // Send the end time of the task to all the waiting tasks.
    send_end_time_waiting_tasks(*task,std::get<END>(schedule_elem),buffers,
        requests);
    this->free_buffers(buffers,requests);
  }

  // Wait for all the buffers to be free before exiting the function.
  while (buffers.size()!=0)
    this->free_buffers(buffers,requests);

  // Update start time.
  end_time = candidate_end_time;
  MPI_Allreduce(MPI_IN_PLACE,&end_time,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,this->mpi_comm);
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::update_schedule_id_map()
{
  schedule_id_map.clear();
  const unsigned int n_tasks(schedule.size());
  for (unsigned int i=0; i<n_tasks; ++i)
    schedule_id_map[std::get<TASK>(schedule[i])->get_local_id()] = i;
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::build_required_tasks_schedule(std::vector<
    std::pair<Task::global_id,discrete_time>> &required_tasks_schedule) const
{
  std::list<discrete_time*> buffers;
  std::list<MPI_Request*> requests;
  // Typedef for the pair (task local id, end time). Local id should be an
  // unsigned int but it will be send through a buffer.
  typedef std::pair<discrete_time,discrete_time> discrete_time_pair;

  // Loop over the tasks in schedule and sort the tasks and their associated end
  // time by waiting processor.
  std::unordered_map<types::subdomain_id,std::vector<discrete_time_pair>> subdomain_time_map;
  std::unordered_map<types::subdomain_id,std::unordered_set<unsigned int>> tasks_done;
  const types::subdomain_id current_subdomain(std::get<TASK> (schedule[0])->get_subdomain_id());
  for (auto const & scheduled_task : schedule)
  {
    const discrete_time task_local_id = static_cast<discrete_time>(
        std::get<TASK>(scheduled_task)->get_local_id());
    const discrete_time task_end_time(std::get<END>(scheduled_task));
    const discrete_time_pair task_time_pair(task_local_id,task_end_time);
    std::vector<Task::subdomain_dof_pair>::const_iterator waiting_subdomain_it(
        std::get<TASK>(scheduled_task)->get_waiting_subdomains_cbegin());
    const std::vector<Task::subdomain_dof_pair>::const_iterator waiting_subdomain_end(
        std::get<TASK>(scheduled_task)->get_waiting_subdomains_cend());
    for (; waiting_subdomain_it!=waiting_subdomain_end; ++waiting_subdomain_it)
    {
      const types::subdomain_id destination(waiting_subdomain_it->first);
      if ((current_subdomain!=destination) &&
          (tasks_done[destination].count(task_local_id)==0))
      {
        subdomain_time_map[destination].push_back(task_time_pair);
        tasks_done[destination].insert(task_local_id);
      }
    }
  }

  // Send the task IDs and the end times to the waiting processors.
  int tag(0);
  std::unordered_map<types::subdomain_id,std::vector<discrete_time_pair>>
    ::const_iterator subdomain_time_map_it(subdomain_time_map.cbegin());
  const std::unordered_map<types::subdomain_id,std::vector<discrete_time_pair>>
    ::const_iterator subdomain_time_map_end(subdomain_time_map.cend());
  for (; subdomain_time_map_it!=subdomain_time_map_end; ++subdomain_time_map_it)
  {
    types::subdomain_id destination(subdomain_time_map_it->first);
    int count(subdomain_time_map_it->second.size()*2);
    discrete_time* buffer = new discrete_time [count];

    std::vector<discrete_time_pair>::const_iterator time_pair_it(
        subdomain_time_map_it->second.cbegin());
    const std::vector<discrete_time_pair>::const_iterator time_pair_end(
        subdomain_time_map_it->second.cend());
    for (unsigned int i=0; time_pair_it!=time_pair_end; ++time_pair_it, i+=2)
    {
      buffer[i] = time_pair_it->first;
      buffer[i+1] = time_pair_it->second;
    }

    buffers.push_back(buffer);
    MPI_Request* request = new MPI_Request;
    requests.push_back(request);

    MPI_Isend(buffer,count,MPI_UNSIGNED_LONG_LONG,destination,tag,this->mpi_comm,request);
  }

  // Loop over the tasks in schedule and sort the tasks by required processor.
  tasks_done.clear();
  std::unordered_map<types::subdomain_id,unsigned int>  subdomain_n_tasks_map;
  for (auto const & scheduled_task : schedule)
  {
    std::vector<Task::task_tuple>::const_iterator required_tasks_it(
        std::get<TASK>(scheduled_task)->get_required_tasks_cbegin());
    const std::vector<Task::task_tuple>::const_iterator required_tasks_end(
        std::get<TASK>(scheduled_task)->get_required_tasks_cend());
    for (; required_tasks_it!=required_tasks_end; ++required_tasks_it)
    {
      const types::subdomain_id source(std::get<0>(*required_tasks_it));
      const unsigned int task_local_id(std::get<1>(*required_tasks_it));
      if ((current_subdomain!=source) && (tasks_done[source].count(task_local_id)==0))
      {
        // This works because the following line is really: 
        // subdomain_n_tasks_map[source] = subdomain_n_tasks_map[source] + 1 and that
        // the first time that subdomain_n_tasks_map[source] is called the value is
        // set to zero.
        subdomain_n_tasks_map[source] += 1;
        tasks_done[source].insert(task_local_id);
      }
    }
  }

  // Receive the data and fill in required_tasks_schedule.
  unsigned int n_received_data(0);
  const unsigned int n_data(subdomain_n_tasks_map.size());
  std::vector<bool> received_data(n_data,false);
  std::unordered_map<types::subdomain_id,unsigned int>
    ::const_iterator subdomain_n_tasks_it(subdomain_n_tasks_map.cbegin());
  const std::unordered_map<types::subdomain_id,unsigned int>
    ::const_iterator subdomain_n_tasks_end(subdomain_n_tasks_map.cend());
  // Loop as long as there are data that have not been received.
  while (n_received_data!=n_data)
  {
    subdomain_n_tasks_it = subdomain_n_tasks_map.cbegin();
    for (unsigned int i=0; subdomain_n_tasks_it!=subdomain_n_tasks_end; ++subdomain_n_tasks_it,++i)
    {
      // Check that we haven't already received the data.
      if (received_data[i]==false)
      {
        types::subdomain_id source(subdomain_n_tasks_it->first);
        int tag(0);
        int flag;

        // Check if we can receive the message
        MPI_Iprobe(source,tag,this->mpi_comm,&flag,MPI_STATUS_IGNORE);

        if (flag==true)
        {
          const int count(2*subdomain_n_tasks_it->second);
          discrete_time* buffer = new discrete_time [count];

          // Receive the message
          MPI_Recv(buffer,count,MPI_UNSIGNED_LONG_LONG,source,tag,this->mpi_comm,
              MPI_STATUS_IGNORE);

          // Unpack the data                          
          for (int i=0; i<count; i+=2)
          {
            const Task::global_id task(source,buffer[i]);
            required_tasks_schedule.push_back(std::pair<Task::global_id,
                discrete_time>(task,buffer[i+1]));
          }

          delete [] buffer;

          received_data[i] = true;
          ++n_received_data;
        }
      }
    }
  }

  // Wait for all the messages to be sent before clearing the buffers and
  // leaving the function.
  std::list<discrete_time*>::iterator buffers_it(buffers.begin());
  std::list<discrete_time*>::iterator buffers_end(buffers.end());
  std::list<MPI_Request*>::iterator requests_it(requests.begin());
  while (buffers_it!=buffers_end)
  {  
    // MPI_Wait deallocate the MPI_Request itself.
    MPI_Wait(*requests_it,MPI_STATUS_IGNORE);
    delete [] *buffers_it;
    buffers_it = buffers.erase(buffers_it);
    requests_it = requests.erase(requests_it);
  }
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::build_waiting_tasks_schedule(std::vector<
    std::pair<Task::global_id,discrete_time>> &waiting_tasks_schedule) const
{
  std::list<discrete_time*> buffers;
  std::list<MPI_Request*> requests;
  // Typedef for the pair (task local id, start time). Local id should be an
  // unsigned int but it will be send through a buffer.
  typedef std::pair<discrete_time,discrete_time> discrete_time_pair;

  // Loop over the tasks in schedule and sort the tasks and their associated
  // start time by required processor.
  std::unordered_map<types::subdomain_id,std::vector<discrete_time_pair>> subdomain_time_map;
  std::unordered_map<types::subdomain_id,std::unordered_set<unsigned int>> tasks_done;
  const types::subdomain_id current_subdomain(std::get<TASK> (schedule[0])->get_subdomain_id());
  for (auto const & scheduled_task : schedule)
  {
    const discrete_time task_local_id = static_cast<discrete_time>(
        std::get<TASK>(scheduled_task)->get_local_id());
    const discrete_time task_start_time(std::get<START>(scheduled_task));
    const discrete_time_pair task_time_pair(task_local_id,task_start_time);
    std::vector<Task::task_tuple>::const_iterator required_tasks_it(
        std::get<TASK>(scheduled_task)->get_required_tasks_cbegin());
    const std::vector<Task::task_tuple>::const_iterator required_tasks_end(
        std::get<TASK>(scheduled_task)->get_required_tasks_cend());
    for (; required_tasks_it!=required_tasks_end; ++required_tasks_it)
    {
      types::subdomain_id destination(std::get<TASK>(*required_tasks_it));
      if ((current_subdomain!=destination) &&
          (tasks_done[destination].count(task_local_id)==0))
      {
        subdomain_time_map[destination].push_back(task_time_pair);
        tasks_done[destination].insert(task_local_id);
      }
    }
  }

  // Send the task IDs and the start times to the required processors.
  int tag(0);
  std::unordered_map<types::subdomain_id,std::vector<discrete_time_pair>>
    ::const_iterator subdomain_time_map_it(subdomain_time_map.cbegin());
  const std::unordered_map<types::subdomain_id,std::vector<discrete_time_pair>>
    ::const_iterator subdomain_time_map_end(subdomain_time_map.cend());
  for (; subdomain_time_map_it!=subdomain_time_map_end; ++subdomain_time_map_it)
  {
    types::subdomain_id destination(subdomain_time_map_it->first);
    int count(subdomain_time_map_it->second.size()*2);
    discrete_time* buffer = new discrete_time [count];

    std::vector<discrete_time_pair>::const_iterator time_pair_it(
        subdomain_time_map_it->second.cbegin());
    const std::vector<discrete_time_pair>::const_iterator time_pair_end(
        subdomain_time_map_it->second.cend());
    for (unsigned int i=0; time_pair_it!=time_pair_end; ++time_pair_it, i+=2)
    {
      buffer[i] = time_pair_it->first;
      buffer[i+1] = time_pair_it->second;
    }

    buffers.push_back(buffer);
    MPI_Request* request = new MPI_Request;
    requests.push_back(request);

    MPI_Isend(buffer,count,MPI_UNSIGNED_LONG_LONG,destination,tag,this->mpi_comm,request);
  }

  // Loop over the tasks in schedule and sort the tasks by waiting processor.
  tasks_done.clear();
  std::unordered_map<types::subdomain_id,unsigned int> subdomain_n_tasks_map;
  for (auto const & scheduled_task : schedule)
  {
    std::vector<Task::task_tuple>::const_iterator waiting_tasks_it(
        std::get<TASK>(scheduled_task)->get_waiting_tasks_cbegin());
    const std::vector<Task::task_tuple>::const_iterator waiting_tasks_end(
        std::get<TASK>(scheduled_task)->get_waiting_tasks_cend());
    for (; waiting_tasks_it!=waiting_tasks_end; ++waiting_tasks_it)
    {
      const types::subdomain_id source(std::get<0>(*waiting_tasks_it));
      const unsigned int task_local_id(std::get<1>(*waiting_tasks_it));
      if ((current_subdomain!=source) && (tasks_done[source].count(task_local_id)==0))
      {
        // This works because the following line is really: 
        // subdomain_n_tasks_map[source] = subdomain_n_tasks_map[source] + 1 and that
        // the first time that subdomain_n_tasks_map[source] is called the value is
        // set to zero.
        subdomain_n_tasks_map[source] += 1;
        tasks_done[source].insert(task_local_id);
      }
    }
  }

  // Receive the data and fill in waiting_tasks_schedule.
  unsigned int n_received_data(0);
  const unsigned int n_data(subdomain_n_tasks_map.size());
  std::vector<bool> received_data(n_data,false);
  std::unordered_map<types::subdomain_id,unsigned int>
    ::const_iterator subdomain_n_tasks_it(subdomain_n_tasks_map.cbegin());
  const std::unordered_map<types::subdomain_id,unsigned int>
    ::const_iterator subdomain_n_tasks_end(subdomain_n_tasks_map.cend());
  // Loop as long as there data that have not been received.
  while (n_received_data!=n_data)
  {
    subdomain_n_tasks_it = subdomain_n_tasks_map.cbegin();
    for (unsigned int i=0; subdomain_n_tasks_it!=subdomain_n_tasks_end; ++subdomain_n_tasks_it,++i)
    {
      // Check that we haven't already received the data.
      if (received_data[i]==false)
      {
        types::subdomain_id source(subdomain_n_tasks_it->first);
        int tag(0);
        int flag;

        // Check if we can receive the message
        MPI_Iprobe(source,tag,this->mpi_comm,&flag,MPI_STATUS_IGNORE);

        if (flag==true)
        {
          const int count(2*subdomain_n_tasks_it->second);
          discrete_time* buffer = new discrete_time [count];

          // Receive the message
          MPI_Recv(buffer,count,MPI_UNSIGNED_LONG_LONG,source,tag,this->mpi_comm,
              MPI_STATUS_IGNORE);

          // Unpack the data
          for (int i=0; i<count; i+=2)
          {
            const Task::global_id task(source,buffer[i]);
            waiting_tasks_schedule.push_back(std::pair<Task::global_id,
                discrete_time>(task,buffer[i+1]));
          }

          delete [] buffer;

          received_data[i] = true;
          ++n_received_data;
        }
      }
    }
  }

  // Wait for all the messages to be sent before clearing the buffers and
  // leaving the function.
  std::list<discrete_time*>::iterator buffers_it(buffers.begin());
  std::list<discrete_time*>::iterator buffers_end(buffers.end());
  std::list<MPI_Request*>::iterator requests_it(requests.begin());
  while (buffers_it!=buffers_end)
  {  
    // MPI_Wait deallocate the MPI_Request itself.
    MPI_Wait(*requests_it,MPI_STATUS_IGNORE);
    delete [] *buffers_it;
    buffers_it = buffers.erase(buffers_it);
    requests_it = requests.erase(requests_it);
  }
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::receive_start_time_waiting_tasks(
    Task const &task,
    std::vector<std::pair<Task::global_id,discrete_time>> &waiting_tasks_schedule) const
{
  const types::subdomain_id current_subdomain(task.get_subdomain_id());

  std::vector<Task::task_tuple>::const_iterator waiting_tasks_it(
      task.get_waiting_tasks_cbegin());
  std::vector<Task::task_tuple>::const_iterator waiting_tasks_end(
      task.get_waiting_tasks_cend());

  for (; waiting_tasks_it!=waiting_tasks_end; ++waiting_tasks_it)
  {
    types::subdomain_id source(std::get<TASK>(*waiting_tasks_it));
    if(source!=current_subdomain)
    {
      unsigned int tag(std::get<1>(*waiting_tasks_it));
      int flag;

      // Check if we can receive a message
      MPI_Iprobe(source,tag,this->mpi_comm,&flag,MPI_STATUS_IGNORE);

      if (flag==true)
      {
        //TODO the messages are awfully short :/
        int count(1);
        discrete_time* buffer = new discrete_time [count];

        // Receive the message
        MPI_Recv(buffer,count,MPI_UNSIGNED_LONG_LONG,source,tag,this->mpi_comm,
            MPI_STATUS_IGNORE);

        waiting_tasks_schedule.push_back(std::pair<Task::global_id,discrete_time>
            (Task::global_id (std::get<0>(*waiting_tasks_it),std::get<1>(*waiting_tasks_it)),
             buffer[0]));

        delete [] buffer;
      }
    }
  }
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::send_start_time_required_tasks(
    Task const &task, discrete_time task_start_time,std::list<discrete_time*> &buffers,
    std::list<MPI_Request*> &requests) const
{
  const types::subdomain_id source(task.get_subdomain_id());
  unsigned int tag(task.get_local_id());

  std::vector<Task::task_tuple>::const_iterator required_tasks_it(
      task.get_required_tasks_cbegin());
  std::vector<Task::task_tuple>::const_iterator required_tasks_end(
      task.get_required_tasks_cend());

  for (; required_tasks_it!=required_tasks_end; ++required_tasks_it)
  {
    types::subdomain_id destination(std::get<0>(*required_tasks_it));

    if (source!=destination)
    {
      //TODO the messages are awfully short :/
      int count(1);
      discrete_time* buffer = new discrete_time [count];
      buffer[0] = task_start_time;
      buffers.push_back(buffer);
      MPI_Request* request = new MPI_Request;
      requests.push_back(request);


      MPI_Isend(buffer,count,MPI_UNSIGNED_LONG_LONG,destination,tag,this->mpi_comm,
          request);
    }
  }
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::receive_end_time_required_tasks(
    Task const &task,
    std::vector<std::pair<Task::global_id,discrete_time>> &required_tasks_schedule) const
{
  const types::subdomain_id current_subdomain(task.get_subdomain_id());

  std::vector<Task::task_tuple>::const_iterator required_tasks_it(
      task.get_required_tasks_cbegin());
  std::vector<Task::task_tuple>::const_iterator required_tasks_end(
      task.get_required_tasks_cend());

  for (; required_tasks_it!=required_tasks_end; ++required_tasks_it)
  {
    types::subdomain_id source(std::get<0>(*required_tasks_it));
    if(source!=current_subdomain)
    {
      unsigned int tag(std::get<1>(*required_tasks_it));
      int flag;

      // Check if we can receive a message
      MPI_Iprobe(source,tag,this->mpi_comm,&flag,MPI_STATUS_IGNORE);

      if (flag==true)
      {
        //TODO the messages are awfully short :/
        int count(1);
        discrete_time* buffer = new discrete_time [count];

        // Receive the message
        MPI_Recv(buffer,count,MPI_UNSIGNED_LONG_LONG,source,tag,this->mpi_comm,
            MPI_STATUS_IGNORE);

        required_tasks_schedule.push_back(std::pair<Task::global_id,discrete_time>
            (Task::global_id(std::get<0>(*required_tasks_it),std::get<1>(*required_tasks_it)),
             buffer[0]));

        delete [] buffer;
      }
    }
  }
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::send_end_time_waiting_tasks(
    Task const &task, discrete_time task_end_time,std::list<discrete_time*> &buffers,
    std::list<MPI_Request*> &requests) const
{
  const types::subdomain_id source(task.get_subdomain_id());
  unsigned int tag(task.get_local_id());

  std::vector<Task::task_tuple>::const_iterator waiting_tasks_it(
      task.get_waiting_tasks_cbegin());
  std::vector<Task::task_tuple>::const_iterator waiting_tasks_end(
      task.get_waiting_tasks_cend());

  for (; waiting_tasks_it!=waiting_tasks_end; ++waiting_tasks_it)
  {
    types::subdomain_id destination(std::get<TASK>(*waiting_tasks_it));

    if (source!=destination)
    {
      //TODO the messages are awfully short :/
      int count(1);
      discrete_time* buffer = new discrete_time [count];
      buffer[0] = task_end_time;
      buffers.push_back(buffer);
      MPI_Request* request = new MPI_Request;
      requests.push_back(request);


      MPI_Isend(buffer,count,MPI_UNSIGNED_LONG_LONG,destination,tag,this->mpi_comm,
          request);
    }
  }
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::clear()
{
  schedule_id_map.clear();
  schedule.clear();

  for (auto & task : this->tasks)
    task.clear();
}



//*****Explicit instantiations*****//
template class CAPPFBScheduler<2,4>;
template class CAPPFBScheduler<2,9>;
template class CAPPFBScheduler<2,16>;
template class CAPPFBScheduler<2,25>;
template class CAPPFBScheduler<2,36>;
template class CAPPFBScheduler<3,8>;
template class CAPPFBScheduler<3,27>;
template class CAPPFBScheduler<3,64>;
template class CAPPFBScheduler<3,125>;
template class CAPPFBScheduler<3,216>;
