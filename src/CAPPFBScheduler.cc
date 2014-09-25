/* Copyright (c) 2014, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "CAPPFBScheduler.hh"


template <int dim,int tensor_dim>
CAPPFBScheduler<dim,tensor_dim>::CAPPFBScheduler(RTQuadrature const* quad,
    Epetra_MpiComm const* comm,unsigned int max_iter,discrete_time tol) :
  Scheduler<dim,tensor_dim>(quad,comm),
  max_iter(max_iter),
  tol(tol),
  start_time(0)
{}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::setup(const unsigned int n_levels,
    std::vector<FECell<dim,tensor_dim>> const* fecell_mesh_ptr,
    std::map<active_cell_iterator,unsigned int> const &cell_to_fecell_map)
{
  Scheduler<dim,tensor_dim>::setup(n_levels,fecell_mesh_ptr,cell_to_fecell_map);

  // TODO:
  // - create initial scheduling
  // - for loop:
  //  - backward iteration (+ save best scheduling)
  //  - forward iteration (+ save best scheduling)
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::start() const
{
  // TODO
}


template <int dim,int tensor_dim>
Task const* const CAPPFBScheduler<dim,tensor_dim>::get_next_task() const
{
  //TODO
  return std::get<0>(schedule[0]);
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::create_initial_scheduling()
{
  this->n_tasks_to_execute = this->tasks.size();
  std::list<discrete_time*> buffers;
  std::list<MPI_Request*> requests;
  unsigned int n_tasks_done(0);
  std::unordered_set<Task::global_id,boost::hash<Task::global_id>> tasks_done;
  std::vector<std::pair<Task::global_id,discrete_time>> required_tasks_schedule;
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
          Task::global_id required_task_id(std::get<0>(*required_task),
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
          for (auto const & scheduled_task : required_tasks_schedule)
          {
            const Task::global_id task_id(std::get<0>(scheduled_task));
            if (task.is_required(task_id)==true)
              if (std::get<1>(scheduled_task)>task_start_time)
                task_start_time = std::get<1>(scheduled_task);
          }

          // Take the maximum of task_start_time and end_time to be sure that
          // the processor is idle long enough.
          task_start_time = std::max(task_start_time,end_time);
          // The number of work units is the number of cells in the patch.
          end_time = task_start_time + task.get_sweep_order_size();
          schedule.push_back(std::tuple<Task*,discrete_time,discrete_time>(&task,
                task_start_time,end_time));
          schedule_id_map[task.get_local_id()] = n_tasks_done;
          tasks_done.insert(task.get_global_id());
          ++n_tasks_done;
          // Tell the other processors that the task has been done, i.e.
          // add it to the tasks_done on other processors and update
          // required_tasks_schedule.
          send_task_done(task,end_time,buffers,requests);
        }
      }
      // Check if some tasks required have been executed by others processors
      // and free usunused buffers.
      receive_tasks_done(tasks_done,required_tasks_schedule);
      this->free_buffers(buffers,requests);
    }
  }
  // Free all the buffers left.
  while (buffers.size()!=0)
    this->free_buffers(buffers,requests);

  // Update start time and end time.
  MPI_Allreduce(MPI_IN_PLACE,&start_time,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,this->mpi_comm);
  MPI_Allreduce(MPI_IN_PLACE,&end_time,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,this->mpi_comm);
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
    std::vector<std::pair<Task::global_id,discrete_time>> &required_tasks_schedule)
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
    types::subdomain_id source(std::get<0>(*global_map_it));
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
      required_tasks_schedule.push_back(std::pair<Task::global_id,discrete_time> 
          (ghost_task,buffer[0]));

      delete [] buffer;
    }
  }
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::backward_iteration()
{
  std::vector<discrete_time> ranks(this->tasks.size());
  compute_backward_ranks(ranks);
  decreasing_rank_sort(ranks);
  backward_scheduling();
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::compute_backward_ranks(std::vector<discrete_time> &ranks) 
{
  // We need the updated end times of the tasks required to execute the tasks
  // locally owned.
  std::vector<std::pair<Task::global_id,discrete_time>> required_tasks_schedule;
  build_required_tasks_schedule(required_tasks_schedule);
  const types::subdomain_id current_subdomain(std::get<0>(schedule[0])->get_subdomain_id());

  for (auto const & scheduled_task : schedule)
  {
    const unsigned int current_task_pos(schedule_id_map[std::get<0>(scheduled_task)->get_local_id()]);
    for (auto required_task=std::get<0>(scheduled_task)->get_required_tasks_cbegin();
        required_task<std::get<0>(scheduled_task)->get_required_tasks_cend(); ++required_task)
    {
      // Search for the time when the latest required task is finished first
      // among the tasks locally owned and after among the tasks owned by other
      // processors.
      Task::global_id required_task_id(std::get<0>(*required_task),
          std::get<1>(*required_task));
      discrete_time candidate_time(0);
      if (required_task_id.first==current_subdomain)
      {
        unsigned int pos(schedule_id_map[required_task_id.second]);
        candidate_time = ranks[pos];
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

      if (candidate_time>ranks[current_task_pos])
        ranks[current_task_pos] = candidate_time;
    }
  }
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::decreasing_rank_sort(std::vector<discrete_time> &ranks)
{
  // Sort the tasks by decreasing rank
  const unsigned int n_tasks(ranks.size());
  for (unsigned int new_pos=0; new_pos<n_tasks; ++new_pos)
  {
    std::vector<discrete_time>::iterator max_rank(std::max_element(
          ranks.begin()+new_pos,ranks.end()));
    const unsigned int old_pos(max_rank-ranks.begin());
    // Put the task earlier in the vector.
    std::swap(ranks[old_pos],ranks[new_pos]);
    std::swap(schedule[old_pos],schedule[new_pos]);
  }

  // Sort the tasks with the same rank
  unsigned int pos(0);
  while (pos<n_tasks)
  {
    unsigned int end_pos(pos+1);
    while (ranks[pos]==ranks[end_pos])
    {
      end_pos += 1;
      if (end_pos==n_tasks)
        break;
    }
    const std::vector<std::tuple<Task*,discrete_time,discrete_time>>
      ::iterator schedule_end_pos(schedule.begin()+end_pos);
    for (unsigned int new_pos=pos; new_pos<end_pos; ++new_pos)
    {
      std::vector<std::tuple<Task*,discrete_time,discrete_time>>
        ::iterator latest_task(schedule.begin()+new_pos);
      std::vector<std::tuple<Task*,discrete_time,discrete_time>>
        ::iterator schedule_it(schedule.begin()+new_pos+1);
      for (; schedule_it!=schedule_end_pos; ++schedule_it)
        if (std::get<2>(*latest_task)<std::get<2>(*schedule_it))
          latest_task = schedule_it;

      const unsigned int old_pos(latest_task-schedule.begin());
      std::swap(schedule[old_pos],schedule[new_pos]);
    }
    pos = end_pos;
  }

  // Need to update schedule_id_map since schedule has been reordered
  update_schedule_id_map();
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::backward_scheduling()
{
  std::list<discrete_time*> buffers;
  std::list<MPI_Request*> requests;

  // Create a vector that represents the discrete time when the processor is
  // idle.
  std::vector<bool> idle_processor(end_time);

  // Tasks that are waiting for the local tasks but are non-local.
  std::vector<std::pair<Task::global_id,discrete_time>> waiting_tasks_schedule;

  const unsigned int n_tasks(schedule.size());
  for (unsigned int i=0; i<n_tasks; ++i)
  {
    discrete_time candidate_time(end_time);
    Task* task(std::get<0>(schedule[i]));
    const unsigned int n_waiting_tasks(task->get_n_waiting_tasks());
    // Check when the waiting tasks are done. Since this sweep is done backward
    // the waiting tasks needs to be "done" before the task can be executed.
    if (n_waiting_tasks!=0)
    {
      bool ready(false);
      while (ready==false)
      {
        ready = true;
        
        // Receive starting time of waiting tasks
        receive_start_time_waiting_tasks(*task,waiting_tasks_schedule);

        for (unsigned int j=0; (j<n_waiting_tasks) && (ready==true); ++j)
        {
          global_id waiting_task_id(task->get_waiting_tasks_global_id(j));
          // If the waiting task is local, search in schedule
          if (std::get<0>(waiting_task_id)==task->get_subdomain_id())
          {
            for (auto const &scheduled_task : schedule)
            {
              if (waiting_task_id==std::get<0>(scheduled_task)->get_global_id())
              {
                if (candidate_time>std::get<1>(scheduled_task))
                  candidate_time = std::get<1>(scheduled_task);
                
                break;
              }
            }
          }
          // If the waiting task is not local, seach in waiting_tasks_schedule
          else
          {
            bool found(false);
            for (auto const & scheduled_waiting_task : waiting_tasks_schedule)
            { 
              if (waiting_task_id==std::get<0>(scheduled_waiting_task))
              {
                if (candidate_time>std::get<1>(scheduled_waiting_task))
                  candidate_time = std::get<1>(scheduled_waiting_task);
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
        if (idle_processor[candidate_time-i]==false)
        {
          candidate_time -= i+1;
          enough_free_time = false;

          break;
        }
      }
    }
    // Mark the processor as busy.
    for (unsigned int i=0; i<execution_time; ++i)
      idle_processor[candidate_time-i] = false;

    // Update the starting and ending time of the task.
    std::get<2>(schedule[i]) = candidate_time;
    std::get<1>(schedule[i]) = candidate_time - execution_time;

    // Send the starting time of the tasks to all the required tasks.
    send_start_time_required_tasks(*task,std::get<1>(schedule[i]),buffers,
        requests);
    this->free_buffers(buffers,requests);
  }

  // Wait for all the buffers to be free before exiting the function.
  while (buffers.size()!=0)
    this->free_buffers(buffers,requests);
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::update_schedule_id_map()
{
  const unsigned int n_tasks(schedule.size());
  for (unsigned int i=0; i<n_tasks; ++i)
    schedule_id_map[std::get<0>(schedule[i])->get_local_id()] = i;
}


template <int dim,int tensor_dim>
void CAPPFBScheduler<dim,tensor_dim>::build_required_tasks_schedule(std::vector<
    std::pair<Task::global_id,discrete_time>> &required_tasks_schedule) const
{
  std::list<discrete_time*> buffers;
  std::list<MPI_Request*> requests;
  // Typedef for the pair (task local id, end time). Local id should be a
  // unsigned int but it will be send through a buffer.
  typedef std::pair<discrete_time,discrete_time> discrete_time_pair;

  // Loop over the tasks in schedule and sort the tasks and their associated end
  // time by waiting processor.
  std::unordered_map<types::subdomain_id,std::unordered_set<discrete_time_pair,
    boost::hash<discrete_time_pair>>> subdomain_time_map;
  for (auto const & scheduled_task : schedule)
  {
    const discrete_time task_local_id = static_cast<discrete_time>(
        std::get<0>(scheduled_task)->get_local_id());
    const discrete_time task_end_time(std::get<2>(scheduled_task));
    const discrete_time_pair task_time(task_local_id,task_end_time);
    std::vector<Task::subdomain_dof_pair>::const_iterator waiting_subdomain_it(
        std::get<0>(scheduled_task)->get_waiting_subdomains_cbegin());
    const std::vector<Task::subdomain_dof_pair>::const_iterator waiting_subdomain_end(
        std::get<0>(scheduled_task)->get_waiting_subdomains_cend());
    for (; waiting_subdomain_it!=waiting_subdomain_end; ++waiting_subdomain_it)
    {
      const types::subdomain_id destination(waiting_subdomain_it->first);
      subdomain_time_map[destination].insert(task_time);
    }
  }

  // Send the task IDs and the end times to the waiting processors.
  int tag(0);
  const types::subdomain_id current_subdomain(std::get<0>
      (schedule[0])->get_subdomain_id());
  std::unordered_map<types::subdomain_id,std::unordered_set<discrete_time_pair,
    boost::hash<discrete_time_pair>>>::const_iterator subdomain_time_map_it(
        subdomain_time_map.cbegin());
  const std::unordered_map<types::subdomain_id,std::unordered_set<discrete_time_pair,
    boost::hash<discrete_time_pair>>>::const_iterator subdomain_time_map_end(
        subdomain_time_map.cend());
  for (; subdomain_time_map_it!=subdomain_time_map_end; ++subdomain_time_map_it)
  {
    types::subdomain_id destination(subdomain_time_map_it->first);
    if (current_subdomain!=destination)
    {
      int count(subdomain_time_map_it->second.size()*2);
      discrete_time* buffer = new discrete_time [count];

      std::unordered_set<discrete_time_pair,boost::hash<discrete_time_pair>>
        ::const_iterator time_pair_it(subdomain_time_map_it->second.cbegin());
      const std::unordered_set<discrete_time_pair,boost::hash<discrete_time_pair>>
        ::const_iterator time_pair_end(subdomain_time_map_it->second.cend());
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
  }

  // Loop over the tasks in schedule and sort the tasks by required processor.
  std::unordered_map<types::subdomain_id,std::unordered_set<unsigned int>> 
    subdomain_task_map;
  for (auto const & scheduled_task : schedule)
  {
    std::vector<Task::task_tuple>::const_iterator required_tasks_it(
        std::get<0>(scheduled_task)->get_required_tasks_cbegin());
    const std::vector<Task::task_tuple>::const_iterator required_tasks_end(
        std::get<0>(scheduled_task)->get_required_tasks_cend());
    for (; required_tasks_it!=required_tasks_end; ++required_tasks_it)
    {
      types::subdomain_id source(std::get<0>(*required_tasks_it));
      subdomain_task_map[source].insert(std::get<1>(*required_tasks_it));
    }
  }

  // Receive the data and fill in required_tasks_schedule.
  unsigned int n_received_data(0);
  const unsigned int n_data(subdomain_task_map.size());
  std::vector<bool> received_data(n_data,false);
  std::unordered_map<types::subdomain_id,std::unordered_set<unsigned int>>
    ::const_iterator subdomain_task_it(subdomain_task_map.cbegin());
  const std::unordered_map<types::subdomain_id,std::unordered_set<unsigned int>>
    ::const_iterator subdomain_task_end(subdomain_task_map.cend());
  // received_data corresponding to the current processor is set to true.
  for (unsigned int i=0; subdomain_task_it!=subdomain_task_end; ++subdomain_task_it, ++i)
    if (current_subdomain==subdomain_task_it->first)
    {
      received_data[i] = true;
      break;
    }
  // Loop as long as there are data that have not been received.
  while (n_received_data!=n_data)
  {
    subdomain_task_it=subdomain_task_map.cbegin();
    for (unsigned int i=0; subdomain_task_it!=subdomain_task_end; ++subdomain_task_it,++i)
    {
      // Check that we haven't already received the data.
      if (received_data[i]==false)
      {
        types::subdomain_id source(subdomain_task_it->first);
        int tag(0);
        int flag;

        // Check if we can receive the message
        MPI_Iprobe(source,tag,this->mpi_comm,&flag,MPI_STATUS_IGNORE);

        if (flag==true)
        {
          const int count(2*subdomain_task_it->second.size());
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
    types::subdomain_id source(std::get<0>(*waiting_tasks_it));
    if(source!=current_subdomain)
    {
      unsigned int tag(std::get<1>(*waiting_tasks_it));
      int flag;

      // Check if we can receive a message
      MPI_Iprobe(source,tag,this->mpi_comm,&flag,MPI_STATUS_IGNORE);

      if (flag==true)
      {
        //TODO the message are awfully short :/
        int count(1);
        discrete_time* buffer = new discrete_time [count];

        // Receive the message
        MPI_Recv(buffer,count,MPI_UNSIGNED_LONG_LONG,source,tag,this->mpi_comm,
            MPI_STATUS_IGNORE);

        global_id waiting_tasks_global_id(std::get<0>(*waiting_tasks_it),
            std::get<1>(*waiting_tasks_it));
        waiting_tasks_schedule.push_back(std::pair<Task::global_id,discrete_time>
            (waiting_tasks_global_id,buffer[0]));

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
      task.get_waiting_tasks_cbegin());
  std::vector<Task::task_tuple>::const_iterator required_tasks_end(
      task.get_waiting_tasks_cend());

  for (; required_tasks_it!=required_tasks_end; ++required_tasks_it)
  {
    types::subdomain_id destination(std::get<0>(*required_tasks_it));

    if (source!=destination)
    {
      //TODO the message are awfully short :/
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
