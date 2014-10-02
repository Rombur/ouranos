/* Copyright (c) 2014, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "Scheduler.hh"

#include "mpi.h"

template <int dim,int tensor_dim>
Scheduler<dim,tensor_dim>::Scheduler(RTQuadrature const* quad,Epetra_MpiComm const* comm) :
  comm(comm),
  mpi_comm(comm->GetMpiComm()),
  quad(quad)
{}


template <int dim,int tensor_dim>
void Scheduler<dim,tensor_dim>::setup(const unsigned int n_levels,
    std::vector<FECell<dim,tensor_dim>> const* fecell_mesh_ptr,
    std::map<active_cell_iterator,unsigned int> const &cell_to_fecell_map)
{
  fecell_mesh = fecell_mesh_ptr;

  // Build patches of cells that will be sweep on.
  std::list<std::list<unsigned int>> cell_patches;
  build_cell_patches(n_levels,cell_to_fecell_map,cell_patches);

  // Compute the sweep ordering.
  compute_sweep_ordering(cell_to_fecell_map,cell_patches);
}


template <int dim,int tensor_dim>
void Scheduler<dim,tensor_dim>::build_cell_patches(const unsigned int n_levels,
    std::map<active_cell_iterator,unsigned int> const &cell_to_fecell_map,
    std::list<std::list<unsigned int>> &cell_patches) const
{
  // Number of cells owned by the current processor.
  const unsigned int n_cells(fecell_mesh->size());

  // The set contains all the cells that are not part of a patch yet.
  std::unordered_set<unsigned int> unused_cells;
  for (unsigned int i=0; i<n_cells; ++i)
    unused_cells.insert(i);

  // Create patches as long as some cells are unused.
  while (unused_cells.size()!=0)
  {
    // Here, we try to get the parent of the current cell and then, we ask for
    // all the children. If all the children are on the same processor, the
    // patch can be accepted and we can try to create a bigger patch using the
    // grandparent cell.

    std::list<unsigned int> current_patch;
    current_patch.push_back(*(unused_cells.begin()));
    unused_cells.erase(unused_cells.begin());
        
    // This is not an active_cell_iterator because we are going "up" the tree of
    // cells and the ancestor cell are not active.
    typename DoFHandler<dim>::cell_iterator current_cell(
        (*fecell_mesh)[*(current_patch.begin())].get_cell());
    std::list<active_cell_iterator> local_active_descendants;
    for (unsigned int i=0; i<n_levels; ++i)
    {
      // Check that the cell has a parent.
      if (current_cell->level()>0)
      {
        current_cell = current_cell->parent();
        // Check that all the active descendants on the same processor. This is
        // done through a recursive function. The function returns true if all
        // the active descendants are locally owned. It also populate a list of
        // all the active descendant of the current cells.
        std::list<active_cell_iterator> active_descendants;
        bool active_descendants_are_local(explore_descendants_tree(current_cell,
              active_descendants));
        if (active_descendants_are_local)
          local_active_descendants = active_descendants;
        else
          break;
      }
    }

    // Add the cells to the patch and take them out of the unused_cells set.
    for (auto const &cell_it : local_active_descendants)
    {
      const unsigned int fecell_local_id(cell_to_fecell_map.at(cell_it));
      if (unused_cells.count(fecell_local_id)==1)
      {
        current_patch.push_back(fecell_local_id);
        unused_cells.erase(fecell_local_id);
      }
    }

    // Add the current patch to others patches.
    cell_patches.push_back(current_patch);
  }
}


template <int dim,int tensor_dim>
bool Scheduler<dim,tensor_dim>::explore_descendants_tree(
    typename DoFHandler<dim>::cell_iterator const &current_cell,
    std::list<active_cell_iterator> &active_descendants) const
{
  bool is_local(true);

  for (unsigned int i=0; i<current_cell->n_children(); ++i)
  {
    is_local = explore_descendants_tree(current_cell->child(i),active_descendants);
    // If one of the descendants is not local, exit the loop.
    if (is_local==false)
      return is_local;
  }

  if (current_cell->active()==true)
  {
    active_descendants.push_back(current_cell);
    is_local = current_cell->is_locally_owned();
  }

  return is_local;
}


template <int dim,int tensor_dim>
void Scheduler<dim,tensor_dim>::compute_sweep_ordering(
    std::map<active_cell_iterator,unsigned int> const &cell_to_fecell_map,
    std::list<std::list<unsigned int>> &cell_patches)
{
  unsigned int task_id(0);

  // Loop on every direction.
  const unsigned int n_dir(quad->get_n_dir());
  for (unsigned int idir=0; idir<n_dir; ++idir)
  {
    // Find the upwind and downwind directions.
    std::vector<unsigned int> downwind_face;
    std::vector<unsigned int> upwind_face;
    Vector<double> const* const omega(quad->get_omega(idir)); 
    for (unsigned int i=0; i<dim; ++i)
    {
      if ((*omega)[i]>0.)
      {
        downwind_face.push_back(2*i+1);
        upwind_face.push_back(2*i);
      }
      else
      {
        downwind_face.push_back(2*i);
        upwind_face.push_back(2*i+1);
      }
    }

    // Do sweep ordering on every patch. Because the patch is assumed to be
    // small, the ordering is not optimized (the candidate cells are all the
    // cells in the patch).
    for (auto const &patch : cell_patches)
    {
      // sweep_order contains the sweep ordering for the cells of a task.
      std::vector<unsigned int> sweep_order;

      // Need a copy of the current patch because we are going to delete some
      // elements
      std::list<unsigned int> current_patch(patch);
      while (current_patch.size()!=0)
      {
        bool accept_cell(true);
        active_cell_iterator cell((*fecell_mesh)[current_patch.front()].get_cell());
        for (unsigned int i=0; i<dim; ++i)
        {
          // If the cell is not at the boundary, the cell has a neighbor upwind.
          if (cell->at_boundary(upwind_face[i])==false)
          {          
            active_cell_iterator upwind_cell(cell->neighbor(upwind_face[i]));
            if (upwind_cell->is_locally_owned())
            {
              // If the upwind cell is in the same patch but has not been used
              // yet, the cell must be rejected.
              if (std::find(current_patch.begin(),current_patch.end(),
                    cell_to_fecell_map.at(upwind_cell))!=current_patch.end())
              {
                accept_cell = false;
                break;
              }
            }
          }
        }
        if (accept_cell==true)
        {
          // The cell is added to the sweep order
          sweep_order.push_back(current_patch.front());
          // The cell is removed to the current patch.
          current_patch.pop_front();
        }
        else
        {
          // The cell is put at the end of the list.
          current_patch.push_back(current_patch.front());
          current_patch.pop_front();
        }
      }

      // Build incomplete_required_tasks, i.e. the required tasks without task_id
      // since it is not known yet.
      std::vector<Task::subdomain_dof_pair> incomplete_required_tasks;
      for (auto const & fecell_id : patch)
      {
        active_cell_iterator cell((*fecell_mesh)[fecell_id].get_cell());
        for (unsigned int i=0; i<dim; ++i)
          if (cell->at_boundary(upwind_face[i])==false)
          {          
            bool upwind_cell_outside_patch(true);

            active_cell_iterator upwind_cell(cell->neighbor(upwind_face[i]));
            if (upwind_cell->is_locally_owned())
              if (std::find(patch.cbegin(),patch.cend(),
                    cell_to_fecell_map.at(upwind_cell))!=patch.cend())
                upwind_cell_outside_patch = false;

            if (upwind_cell_outside_patch==true)
            {
              std::vector<types::global_dof_index> dof_indices(tensor_dim);
              upwind_cell->get_dof_indices(dof_indices);
              Task::subdomain_dof_pair subdomain_dof_pair(upwind_cell->subdomain_id(),
                  dof_indices);
              incomplete_required_tasks.push_back(subdomain_dof_pair);
            }
          }
      }

      // All the cells of a patch are on the same processor.
      types::subdomain_id subdomain_id((*fecell_mesh)[sweep_order[0]].get_cell()->subdomain_id());

      tasks.push_back(Task(idir,task_id,subdomain_id,sweep_order,
            incomplete_required_tasks));
      ++task_id;
    }
  }

  // Check that each processor has at least one task to execute.
  Assert(tasks.size()!=0,ExcMessage("One processor has no task to execute."));

  // Build the maps of tasks waiting for a given task to be done
  build_waiting_tasks_maps();

  // Build the maps of task required for a given task to start
  build_required_tasks_maps();
 
  // Loop over the tasks and create the necessary remaining maps and delete
  // the ones that are not necessary anymore.
  for (unsigned int i=0; i<tasks.size(); ++i)
    tasks[i].finalize_maps();

  // Build an unique map per processor which given a required task by one of
  // the task owned by the processor and a dof return the position this dof in
  // the MPI buffer.
  build_global_required_tasks();

  build_local_tasks_map();
}


template <int dim,int tensor_dim>
void Scheduler<dim,tensor_dim>::build_waiting_tasks_maps()
{
  const unsigned int n_proc(comm->NumProc());
  const unsigned int n_tasks(tasks.size());

  // Send the number of elements that each processor will receive
  int* send_n_dofs_buffer = new int [n_proc];
  int* recv_n_dofs_buffer = new int [n_proc];
  const int send_n_dofs_count(1);
  const int recv_n_dofs_count(1);
  std::fill(send_n_dofs_buffer,send_n_dofs_buffer+n_proc,0);
  
  for (unsigned int i=0; i<n_tasks; ++i)
  {
    const unsigned int n_required_tasks(tasks[i].get_incomplete_n_required_tasks());
    for (unsigned int j=0; j<n_required_tasks; ++j)
      send_n_dofs_buffer[tasks[i].get_incomplete_subdomain_id(j)] += 
        tasks[i].get_incomplete_n_dofs(j)+3;
  }

  MPI_Alltoall(send_n_dofs_buffer,send_n_dofs_count,MPI_INT,recv_n_dofs_buffer,
      recv_n_dofs_count,MPI_INT,mpi_comm);

  // Send the task IDs, the directions, and the dofs. The buffer looks like: 
  // [task_id,idir,n_dofs,dof,dof,dof,task_id,idir,n_dofs,dof,dof,...]
  int* send_dof_disps = new int [n_proc];
  int* recv_dof_disps = new int [n_proc];
  send_dof_disps[0] = 0; 
  recv_dof_disps[0] = 0;
  std::vector<int> offset(n_proc);
  for (unsigned int i=1; i<n_proc; ++i)
  {
    send_dof_disps[i] = send_dof_disps[i-1] + send_n_dofs_buffer[i-1];
    recv_dof_disps[i] = recv_dof_disps[i-1] + recv_n_dofs_buffer[i-1];
    offset[i] = send_dof_disps[i];
  }
  
  const unsigned int recv_dof_buffer_size(recv_dof_disps[n_proc-1]+
      recv_n_dofs_buffer[n_proc-1]);
  types::global_dof_index* send_dof_buffer = 
    new types::global_dof_index [send_dof_disps[n_proc-1]+
    send_n_dofs_buffer[n_proc-1]];
  types::global_dof_index* recv_dof_buffer = 
    new types::global_dof_index [recv_dof_buffer_size];
  for (unsigned int i=0; i<n_tasks; ++i)
  {
    const unsigned int n_required_tasks(tasks[i].get_incomplete_n_required_tasks());
    for (unsigned int j=0; j<n_required_tasks; ++j)
    {
      const unsigned int subdomain_id(tasks[i].get_incomplete_subdomain_id(j));
      const unsigned int current_offset(offset[subdomain_id]);
      const unsigned int n_dofs_task(tasks[i].get_incomplete_n_dofs(j));
      std::vector<types::global_dof_index> const* task_dof(
          tasks[i].get_incomplete_dofs(j));
      send_dof_buffer[current_offset] = static_cast<types::global_dof_index>(
          tasks[i].get_local_id());
      send_dof_buffer[current_offset+1] = static_cast<types::global_dof_index>(
          tasks[i].get_idir());
      send_dof_buffer[current_offset+2] = static_cast<types::global_dof_index>(
          n_dofs_task);
      for (unsigned int k=0; k<n_dofs_task; ++k)
        send_dof_buffer[current_offset+k+3] = (*task_dof)[k];   
      offset[subdomain_id] += n_dofs_task+3;
    }
  }

  MPI_Alltoallv(send_dof_buffer,send_n_dofs_buffer,send_dof_disps,
      DEAL_II_DOF_INDEX_MPI_TYPE,recv_dof_buffer,recv_n_dofs_buffer,recv_dof_disps,
      DEAL_II_DOF_INDEX_MPI_TYPE,mpi_comm);           

  // Build the extended recv_dof_disps, i.e., recv_dof_disps plus
  // recv_dof_buffer_size
  int* recv_dof_disps_x = new int [n_proc+1];
  for (unsigned int i=0; i<n_proc; ++i)
     recv_dof_disps_x[i] = recv_dof_disps[i];
  recv_dof_disps_x[n_proc] = recv_dof_buffer_size;
  
  // Now every processor can fill in the waiting_tasks map of each task
  for (unsigned int i=0; i<n_tasks; ++i)
    build_local_waiting_tasks_map(tasks[i],recv_dof_buffer,recv_dof_disps_x,
        recv_dof_buffer_size);

  delete [] recv_dof_disps_x;
  delete [] recv_dof_buffer;
  delete [] send_dof_buffer;
  delete [] recv_dof_disps;
  delete [] send_dof_disps;
  delete [] recv_n_dofs_buffer;
  delete [] send_n_dofs_buffer;
  recv_dof_disps_x = nullptr;    
  recv_dof_buffer = nullptr;
  send_dof_buffer = nullptr;   
  recv_dof_disps = nullptr;    
  send_dof_disps = nullptr;    
  recv_n_dofs_buffer = nullptr;
  send_n_dofs_buffer = nullptr;
}


template <int dim,int tensor_dim>
void Scheduler<dim,tensor_dim>::build_local_waiting_tasks_map(Task &task,
    types::global_dof_index* recv_dof_buffer,int* recv_dof_disps_x,
    const unsigned int recv_dof_buffer_size)
{
  // Get the dofs associated to the current task.
  std::unordered_set<types::global_dof_index> local_dof_indices(get_task_local_dof_indices(task));

  // Build the waiting_tasks map.
  unsigned int i(0);
  unsigned int subdomain_id(0);
  unsigned int next_subdomain_disps(recv_dof_disps_x[1]);
  while (i<recv_dof_buffer_size)
  {
    const unsigned int task_id(recv_dof_buffer[i]);
    const unsigned int idir(recv_dof_buffer[i+1]);
    const unsigned int n_dofs(recv_dof_buffer[i+2]);
    // Increment the subdomain ID of the waiting task if necessary.
    while (i==next_subdomain_disps)
    {
      ++subdomain_id;
      next_subdomain_disps = recv_dof_disps_x[subdomain_id+1];
    }
    // Search for dofs present in recv_dof_buffer and local_dof_indices
    if (idir==task.get_idir())
    {
      std::vector<types::global_dof_index> dofs;
      for (unsigned int j=0; j<n_dofs; ++j)
      {
        // If the dof in recv_dof_buffer is in local_dof_indices, the dof is
        // added in the waiting map  .
        if (local_dof_indices.count(recv_dof_buffer[i+3+j])==1)
          dofs.push_back(recv_dof_buffer[i+3+j]);
      }
      if (dofs.size()!=0)
      {
        task.add_to_waiting_tasks(Task::task_tuple (subdomain_id,task_id,dofs));
        task.add_to_waiting_subdomains(Task::subdomain_dof_pair (subdomain_id,dofs));
      }
    }
    i += n_dofs+3;
  }

  // Compress the waiting_tasks, i.e., suppress duplicated tasks that have the
  // same subdomain_id and task_id and accumulate the dofs in the unique tasks.
  task.compress_waiting_tasks();

  // Sort the dofs associated to waiting processors (subdomains) and suppress
  // duplicates.
  task.compress_waiting_subdomains();
}  


template <int dim,int tensor_dim>
void Scheduler<dim,tensor_dim>::build_required_tasks_maps()
{  
  const unsigned int n_proc(comm->NumProc());
  const unsigned int n_tasks(tasks.size());

  // Send the number of elements that each processor will receive
  int* send_n_dofs_buffer = new int [n_proc];
  int* recv_n_dofs_buffer = new int [n_proc];
  const int send_n_dofs_count(1);
  const int recv_n_dofs_count(1);
  std::fill(send_n_dofs_buffer,send_n_dofs_buffer+n_proc,0);

  for (unsigned int i=0; i<n_tasks; ++i)
  {
    const unsigned int n_waiting_tasks(tasks[i].get_n_waiting_tasks());
    for (unsigned int j=0; j<n_waiting_tasks; ++j)
      send_n_dofs_buffer[tasks[i].get_waiting_tasks_subdomain_id(j)] += 
        tasks[i].get_waiting_tasks_n_dofs(j)+3;
  }

  MPI_Alltoall(send_n_dofs_buffer,send_n_dofs_count,MPI_INT,recv_n_dofs_buffer,
      recv_n_dofs_count,MPI_INT,mpi_comm);

  // Send the dofs and the task IDs. The buffer looks like: 
  // [task_id sender,task_id recv,n_dofs,dof,dof,dof,...]
  int* send_dof_disps = new int [n_proc];
  int* recv_dof_disps = new int [n_proc];
  send_dof_disps[0] = 0; 
  recv_dof_disps[0] = 0;
  std::vector<int> offset(n_proc);
  for (unsigned int i=1; i<n_proc; ++i)
  {
    send_dof_disps[i] = send_dof_disps[i-1] + send_n_dofs_buffer[i-1];
    recv_dof_disps[i] = recv_dof_disps[i-1] + recv_n_dofs_buffer[i-1];
    offset[i] = send_dof_disps[i];
  }
 
  const unsigned int recv_dof_buffer_size(recv_dof_disps[n_proc-1]+
      recv_n_dofs_buffer[n_proc-1]);
  types::global_dof_index* send_dof_buffer = 
    new types::global_dof_index [send_dof_disps[n_proc-1]+
    send_n_dofs_buffer[n_proc-1]];
  types::global_dof_index* recv_dof_buffer = 
    new types::global_dof_index [recv_dof_buffer_size];

  for (unsigned int i=0; i<n_tasks; ++i)
  {
    const unsigned int n_waiting_tasks(tasks[i].get_n_waiting_tasks());
    for (unsigned int j=0; j<n_waiting_tasks; ++j)
    {
      const types::subdomain_id subdomain_id(tasks[i].get_waiting_tasks_subdomain_id(
            j));
      const unsigned int current_offset(offset[subdomain_id]);
      const unsigned int n_dofs_task(tasks[i].get_waiting_tasks_n_dofs(j));
      std::vector<types::global_dof_index> const* task_dof(
          tasks[i].get_waiting_tasks_dofs(j));
      send_dof_buffer[current_offset] = static_cast<types::global_dof_index>(
          tasks[i].get_local_id());
      send_dof_buffer[current_offset+1] = static_cast<types::global_dof_index>(
          tasks[i].get_waiting_tasks_local_id(j));
      send_dof_buffer[current_offset+2] = static_cast<types::global_dof_index>(
          n_dofs_task);
      for (unsigned int k=0; k<n_dofs_task; ++k)
        send_dof_buffer[current_offset+k+3] = (*task_dof)[k];
      offset[subdomain_id] += n_dofs_task+3;
    }  
  }

  MPI_Alltoallv(send_dof_buffer,send_n_dofs_buffer,send_dof_disps,
      DEAL_II_DOF_INDEX_MPI_TYPE,recv_dof_buffer,recv_n_dofs_buffer,recv_dof_disps,
      DEAL_II_DOF_INDEX_MPI_TYPE,mpi_comm);
  
  // Build the extended recv_dof_disps, i.e., recv_dof_disps plus
  // recv_dof_buffer_size
  int* recv_dof_disps_x = new int [n_proc+1];
  for (unsigned int i=0; i<n_proc; ++i)
     recv_dof_disps_x[i] = recv_dof_disps[i];
  recv_dof_disps_x[n_proc] = recv_dof_buffer_size;
  
  // Now every processor can fill in the required_tasks map of each task
  for (unsigned int i=0; i<n_tasks; ++i)
    build_local_required_tasks_map(tasks[i],recv_dof_buffer,recv_dof_disps_x,
        recv_dof_buffer_size);

  delete [] recv_dof_disps_x;
  delete [] recv_dof_buffer;
  delete [] send_dof_buffer;
  delete [] recv_dof_disps;
  delete [] send_dof_disps;
  delete [] recv_n_dofs_buffer;
  delete [] send_n_dofs_buffer;
  recv_dof_disps_x = nullptr;    
  recv_dof_buffer = nullptr;
  send_dof_buffer = nullptr;   
  recv_dof_disps = nullptr;    
  send_dof_disps = nullptr;    
  recv_n_dofs_buffer = nullptr;
  send_n_dofs_buffer = nullptr;
}


template <int dim,int tensor_dim>
void Scheduler<dim,tensor_dim>::build_local_required_tasks_map(Task &task,
    types::global_dof_index* recv_dof_buffer,int* recv_dof_disps_x,
    const unsigned int recv_dof_buffer_size)
{
  const unsigned int current_task_id(task.get_local_id());

  // Build the required_tasks map
  unsigned int subdomain_id(0);
  unsigned int next_subdomain_disps(recv_dof_disps_x[1]);
  for (unsigned int i=0; i<recv_dof_buffer_size;)
  {
    const unsigned int sender_task_id(recv_dof_buffer[i]);
    const unsigned int recv_task_id(recv_dof_buffer[i+1]);
    const unsigned int n_dofs(recv_dof_buffer[i+2]);
    // Increment the subdomain ID of the required task if necessary
    while (i==next_subdomain_disps)
    {
      ++subdomain_id;
      next_subdomain_disps = recv_dof_disps_x[subdomain_id+1];
    }
    // If the receiver task is the current task, add the dofs to required_tasks_map
    if (recv_task_id==current_task_id)
      task.add_to_required_tasks(subdomain_id,sender_task_id,recv_dof_buffer,i+3,n_dofs);
    
    i += n_dofs+3;
  }
}


template <int dim,int tensor_dim>
void Scheduler<dim,tensor_dim>::build_global_required_tasks()
{
  const unsigned int n_tasks(tasks.size());
  AssertIndexRange(0,n_tasks);
  const types::subdomain_id subdomain_id(tasks[0].get_subdomain_id());
  std::unordered_map<Task::global_id,std::vector<types::global_dof_index>,
    boost::hash<Task::global_id>> tmp_map;
  // Loop over all the required_tasks and create a temporary map with sorted
  // dofs
  for (unsigned int i=0; i<n_tasks; ++i)
  {
    std::vector<Task::task_tuple>::const_iterator map_it(tasks[i].get_required_tasks_cbegin());
    std::vector<Task::task_tuple>::const_iterator map_end(tasks[i].get_required_tasks_cend());
    for (; map_it!=map_end; ++map_it)
    {
      // Check that the required task is not owned by the processor
      if (std::get<0>(*map_it)!=subdomain_id)
      {
        Task::global_id current_task(std::get<0>(*map_it),std::get<1>(*map_it));
        Assert(std::is_sorted(tmp_map[current_task].begin(),tmp_map[current_task].end()),
            ExcMessage("The temporary map tmp_map is not sorted."));
        Assert(std::is_sorted(std::get<2>(*map_it).begin(),std::get<2>(*map_it).end()),
            ExcMessage("The vector of dofs of the required task is not sorted."));

        std::vector<types::global_dof_index> sorted_union(
            tmp_map[current_task].size()+std::get<2>(*map_it).size());
        std::vector<types::global_dof_index>::iterator vector_it;
        vector_it = std::set_union(tmp_map[current_task].begin(),tmp_map[current_task].end(),
            std::get<2>(*map_it).begin(),std::get<2>(*map_it).end(),sorted_union.begin());
        sorted_union.resize(vector_it-sorted_union.begin());
        tmp_map[current_task] = sorted_union;

        // Create the ghost_required_tasks (current_task is a task on an other
        // processor and i is the position in the local tasks vector)
        ghost_required_tasks[current_task].push_back(i);
      }
    }
  }

  // Build the global_required_tasks map, by looping over the tmp_map and
  // adding the position of the dofs in the MPI messages
  std::unordered_map<Task::global_id,std::vector<types::global_dof_index>,
    boost::hash<Task::global_id>>::iterator tmp_map_it(tmp_map.begin());
  std::unordered_map<Task::global_id,std::vector<types::global_dof_index>,
    boost::hash<Task::global_id>>::iterator 
      tmp_map_end(tmp_map.end());
  for (; tmp_map_it!=tmp_map_end; ++tmp_map_it)
  {
    const unsigned int i_max(tmp_map_it->second.size());
    global_required_tasks.push_back(std::tuple<types::subdomain_id,unsigned int,
            std::unordered_map<types::global_dof_index,unsigned int>> (
              tmp_map_it->first.first,tmp_map_it->first.second,
              std::unordered_map<types::global_dof_index,unsigned int> ()));
    for (unsigned int i=0; i<i_max; ++i)
      std::get<2>(global_required_tasks.back())[tmp_map_it->second[i]] = i;
  }
}


template <int dim,int tensor_dim>
void Scheduler<dim,tensor_dim>::build_local_tasks_map()
{
  const unsigned int n_tasks(tasks.size());
  for (unsigned int i=0; i<n_tasks; ++i)
    local_tasks_map[tasks[i].get_local_id()] = i;
}


template <int dim,int tensor_dim>
std::unordered_set<types::global_dof_index> 
Scheduler<dim,tensor_dim>::get_task_local_dof_indices(Task &task)
{
  std::unordered_set<types::global_dof_index> local_dof_indices;
  std::vector<unsigned int> const* sweep_order(task.get_sweep_order());
  const unsigned int sweep_order_size(task.get_sweep_order_size());
  // Loop over the cells in the sweep associated to this task
  for (unsigned int i=0; i<sweep_order_size; ++i)
  {
    // Copy the dofs associated to the cell to local_dof_indices
    active_cell_iterator cell((*fecell_mesh)[(*sweep_order)[i]].get_cell());
    std::vector<types::global_dof_index> cell_dof_indices(tensor_dim);
    cell->get_dof_indices(cell_dof_indices);
    for (unsigned int j=0; j<tensor_dim; ++j)
      local_dof_indices.insert(cell_dof_indices[j]);
  }

  return local_dof_indices;
}


template <int dim,int tensor_dim>
void Scheduler<dim,tensor_dim>::send_angular_flux(Task const &task,
    std::list<double*> &buffers,std::list<MPI_Request*> &requests) const
{
  std::vector<Task::subdomain_dof_pair>::const_iterator map_it(
      task.get_waiting_subdomains_cbegin());
  std::vector<Task::subdomain_dof_pair>::const_iterator map_end(
      task.get_waiting_subdomains_cend());

  // Loop over all the waiting processors
  types::subdomain_id source(task.get_subdomain_id());
  unsigned int tag(task.get_local_id());
  for (; map_it!=map_end; ++map_it)
  {
    types::subdomain_id destination(map_it->first);
    // If source and destination processors are different, MPI is used
    if (source!=destination)
    {
      int count(map_it->second.size());
      double* buffer = new double [count];
      // Copy the requested dofs to the buffer
      for (int i=0; i<count; ++i)
        buffer[i] = task.get_required_angular_flux(map_it->second[i]);
      buffers.push_back(buffer);
      MPI_Request* request = new MPI_Request;
      requests.push_back(request);

      MPI_Isend(buffer,count,MPI_DOUBLE,destination,tag,mpi_comm,request);
    }
    else
    {
      // if source and destination processors are the same, loop over a subset
      // of the waiting_tasks and set the required dofs if necessary
      std::vector<std::pair<unsigned int,std::vector<types::global_dof_index>>>
        ::const_iterator waiting_tasks_it(task.get_local_waiting_tasks_cbegin());
      std::vector<std::pair<unsigned int,std::vector<types::global_dof_index>>>
        ::const_iterator waiting_tasks_cend(task.get_local_waiting_tasks_cend());
      for (; waiting_tasks_it!=waiting_tasks_cend; ++waiting_tasks_it)
      {
        const unsigned int local_pos(local_tasks_map[waiting_tasks_it->first]);
        // Check that the task is not the one that was just executed.
        if (local_pos!=tag)
        {
          std::vector<types::global_dof_index>::const_iterator dofs_it(
              waiting_tasks_it->second.cbegin());
          std::vector<types::global_dof_index>::const_iterator dofs_end(
              waiting_tasks_it->second.cend());
          for (; dofs_it!=dofs_end; ++dofs_it)
            tasks[local_pos].set_required_dof(*dofs_it,task.get_required_angular_flux(*dofs_it));
        }
      }
    }
  }
}


template <int dim,int tensor_dim>
void Scheduler<dim,tensor_dim>::receive_angular_flux() const 
{
  // Loop on the global_required_tasks map
  std::vector<std::tuple<types::subdomain_id,unsigned int,
    std::unordered_map<types::global_dof_index,unsigned int>>>::iterator
      global_map_it(global_required_tasks.begin());
  std::vector<std::tuple<types::subdomain_id,unsigned int,
    std::unordered_map<types::global_dof_index,unsigned int>>>::iterator
      global_map_end(global_required_tasks.end());
  for (; global_map_it!=global_map_end; ++global_map_it)
  {
    types::subdomain_id source(std::get<0>(*global_map_it));
    unsigned int tag(std::get<1>(*global_map_it));
    int flag;

    // Check if we can receive a message
    MPI_Iprobe(source,tag,mpi_comm,&flag,MPI_STATUS_IGNORE);

    if (flag==true)
    {
      const Task::global_id ghost_task(source,tag);
      int count(std::get<2>(*global_map_it).size());
      double* buffer = new double [count];

      // Receive the message
      MPI_Recv(buffer,count,MPI_DOUBLE,source,tag,mpi_comm,MPI_STATUS_IGNORE);

      // Set the required dofs by looping on the tasks that are waiting for
      // the ghost cell
      std::vector<unsigned int>::const_iterator required_tasks_it(
          ghost_required_tasks[ghost_task].cbegin());
      std::vector<unsigned int>::const_iterator required_tasks_end(
          ghost_required_tasks[ghost_task].cend());
      for (; required_tasks_it!=required_tasks_end; ++required_tasks_it)
      {
        std::vector<types::global_dof_index> const* const required_dofs(
            tasks[*required_tasks_it].get_required_dofs(source,tag));
        const unsigned int required_dofs_size(required_dofs->size());
        for (unsigned int j=0; j<required_dofs_size; ++j)
        {
          const unsigned int required_dof((*required_dofs)[j]);
          const unsigned int buffer_pos(std::get<2>(*global_map_it)[required_dof]);
          tasks[*required_tasks_it].set_required_dof(required_dof,buffer[buffer_pos]);
        }
      }

      delete [] buffer;
    }
  }      
}


//*****Explicit instantiations*****//
template class Scheduler<2,4>;
template class Scheduler<2,9>;
template class Scheduler<2,16>;
template class Scheduler<2,25>;
template class Scheduler<2,36>;
template class Scheduler<3,8>;
template class Scheduler<3,27>;
template class Scheduler<3,64>;
template class Scheduler<3,125>;
template class Scheduler<3,216>;
