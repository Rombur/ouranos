/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "RadiativeTransfer.hh"

#include <iostream>

template <int dim,int tensor_dim>
RadiativeTransfer<dim,tensor_dim>::RadiativeTransfer(FE_DGQ<dim>* fe,
    parallel::distributed::Triangulation<dim>* triangulation,
    DoFHandler<dim>* dof_handler,Parameters* parameters,RTQuadrature* quad,
    RTMaterialProperties* material_properties,Epetra_MpiComm const* comm,
    Epetra_Map const* map) :
  n_mom(quad->get_n_mom()),
  group(0),
  n_groups(1),
  comm(comm),
  map(map),
  fe(fe),
  triangulation(triangulation),
  dof_handler(dof_handler),
  parameters(parameters),
  quad(quad),
  material_properties(material_properties)
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
  typename DoFHandler<dim>::active_cell_iterator cell(dof_handler->begin_active()),
           end_cell(dof_handler->end());
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

  for (; cell<end_cell; ++cell)
    if (cell->is_locally_owned())
    {
      FECell<dim,tensor_dim> fecell(n_quad_points,n_face_quad_points,
          fe_values,fe_face_values,fe_neighbor_face_values,cell,end_cell);
      fecell_mesh.push_back(fecell);
    }
 
  // Compute the sweep ordering
  compute_sweep_ordering();
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::compute_sweep_ordering()
{
  // First find the cells on the ''boundary'' of the processor
  std::vector<std::set<unsigned int> > boundary_cells(2*dim);
  const unsigned int fec_mesh_size(fecell_mesh.size());
  for (unsigned int i=0; i<fec_mesh_size; ++i)
  {
    typename DoFHandler<dim>::active_cell_iterator cell(fecell_mesh[i].get_cell());
    for (unsigned int face=0; face<2*dim; ++face)
    {
      // Check if the face is on the boundary of the domain
      if (cell->at_boundary(face)==true)
        boundary_cells[face].insert(i);
      else
        if (cell->neighbor(face)->is_ghost()==true)
          boundary_cells[face].insert(i);
    }
  }         

  unsigned int task_id(0);
  const unsigned int n_dir(quad->get_n_dir());
  for (unsigned int idir=0; idir<n_dir; ++idir)
  { 
    // Cells already used in a task
    std::set<typename DoFHandler<dim>::active_cell_iterator> used_cells;
    // Candidate cells for the sweep order
    std::list<unsigned int> candidate_cells;
    
    // Find the cells on the boundary
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
    candidate_cells.resize(boundary_cells[upwind_face[0]].size()+
        boundary_cells[upwind_face[1]].size());
    std::list<unsigned int>::iterator list_it;
    list_it = std::set_union(boundary_cells[upwind_face[0]].begin(),
        boundary_cells[upwind_face[0]].end(),
        boundary_cells[upwind_face[1]].begin(),
        boundary_cells[upwind_face[1]].end(),candidate_cells.begin());
    // Resize candidate_cells
    unsigned int new_size(0);
    std::list<unsigned int>::iterator tmp_it(candidate_cells.begin());
    for (; tmp_it!=list_it; ++tmp_it)
      ++new_size;
    candidate_cells.resize(new_size);
    if (dim==3)
    {
      std::list<unsigned int> tmp_list(candidate_cells);
      candidate_cells.resize(tmp_list.size()+boundary_cells[upwind_face[2]].size());
      list_it = std::set_union(tmp_list.begin(),tmp_list.end(),
          boundary_cells[upwind_face[2]].begin(),
          boundary_cells[upwind_face[2]].end(),candidate_cells.begin());
      // Resize candidate_cells
      new_size = 0;
      tmp_it = candidate_cells.begin();
      for (; tmp_it!=list_it; ++list_it)
        ++new_size;
      candidate_cells.resize(new_size);
    }

    // Build the sweep order
    while (candidate_cells.size()!=0)
    {
      // For now there is only one cell per task, sweep_order contains only
      // one element
      std::vector<unsigned int> sweep_order;
      // Required task but missing task_id
      std::vector<std::pair<types::subdomain_id,std::vector<types::global_dof_index>>>
          incomplete_required_tasks;

      typename DoFHandler<dim>::active_cell_iterator cell(
          fecell_mesh[candidate_cells.front()].get_cell());
      for (unsigned int i=0; i<dim; ++i)
      {
        bool other_task(cell->at_boundary(upwind_face[i]) ? false : true);
        if (other_task==true)
        {
          typename DoFHandler<dim>::active_cell_iterator neighbor_cell(
              cell->neighbor(upwind_face[i]));
          std::vector<types::global_dof_index> dof_indices(tensor_dim);
          neighbor_cell->get_dof_indices(dof_indices);
          std::pair<types::subdomain_id,std::vector<types::global_dof_index>>
            subdomain_dof_pair(neighbor_cell->subdomain_id(),dof_indices);
          incomplete_required_tasks.push_back(subdomain_dof_pair);
        }
      }             
      types::subdomain_id subdomain_id(cell->subdomain_id());
      // The cell is added to the sweep order
      sweep_order.push_back(candidate_cells.front());
      // The cell is removed from candidate_cells and added to used_cells
      used_cells.insert(cell);
      candidate_cells.pop_front();
      // Add the candidate cells
      for (unsigned int i=0; i<dim; ++i)
      {
        if (cell->at_boundary(downwind_face[i])==false)
        {
          typename DoFHandler<dim>::active_cell_iterator neighbor_cell(
              cell->neighbor(downwind_face[i]));
          if ((neighbor_cell->is_locally_owned()==true) && 
              (used_cells.count(neighbor_cell)==0))
          {
            const unsigned fecell_mesh_size(fecell_mesh.size());
            for (unsigned int j=0; j<fecell_mesh_size; ++j)
            {
              if ((fecell_mesh[j].get_cell()==neighbor_cell) &&
                  (std::find(candidate_cells.begin(),candidate_cells.end(),j)==
                   candidate_cells.end()))
              {
                candidate_cells.push_back(j);
                break;
              }
            }
          }
        }
      }

      tasks.push_back(Task(idir,task_id,subdomain_id,sweep_order,
            incomplete_required_tasks));
      ++task_id;
    }
  }

  // Build the maps of tasks waiting for a given task to be done
  build_waiting_tasks_maps();

  // Build the maps of task required for a given task to start
  build_required_tasks_maps();
 
  // Loop over the tasks and clear the temporary data that are not needed
  // anymore
  for (unsigned int i=0; i<tasks.size(); ++i)
    tasks[i].clear_temporary_data();

  // Build an unique map per processor which given a required task by one of
  // the task owned by the processor and a dof return the position this dof in
  // the MPI buffer.
  build_global_required_tasks();
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::build_waiting_tasks_maps()
{
  const unsigned int n_proc(comm->NumProc());
  const unsigned int n_tasks(tasks.size());
  MPI_Comm mpi_comm(comm->GetMpiComm());

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
          tasks[i].get_id());
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
void RadiativeTransfer<dim,tensor_dim>::build_local_waiting_tasks_map(Task &task,
    types::global_dof_index* recv_dof_buffer,int* recv_dof_disps_x,
    const unsigned int recv_dof_buffer_size)
{
  // Get the dofs associated to the current task
  const unsigned int sweep_order_size(task.get_sweep_order_size());
  std::vector<types::global_dof_index> local_dof_indices(sweep_order_size*tensor_dim);
  get_task_local_dof_indices(task,local_dof_indices);

  // Build the waiting_tasks map
  unsigned int subdomain_id(0);
  unsigned int next_subdomain_disps(recv_dof_disps_x[1]);
  for (unsigned int i=0; i<recv_dof_buffer_size;)
  {
    const unsigned int task_id(recv_dof_buffer[i]);
    const unsigned int idir(recv_dof_buffer[i+1]);
    const unsigned int n_dofs(recv_dof_buffer[i+2]);
    // Increment the subdomain ID of the waiting task if necessary
    while (i==next_subdomain_disps)
    {
      ++subdomain_id;
      next_subdomain_disps = recv_dof_disps_x[subdomain_id+1];
    }
    // Search for dofs present in recv_dof_buffer and local_dof_indices
    if (idir==task.get_idir())
    {
      for (unsigned int j=0; j<n_dofs; ++j)
      {
        // If the dof in recv_dof_buffer is in local_dof_indices, the dof is
        // added in the waiting map  
        if (std::find(local_dof_indices.begin(),local_dof_indices.end(),
              recv_dof_buffer[i+3+j])!=local_dof_indices.end())
        {
          std::pair<types::subdomain_id,unsigned int> subdomain_task_pair(
              subdomain_id,task_id);
          task.add_to_waiting_tasks(subdomain_task_pair,recv_dof_buffer[i+3+j]);
          task.add_to_waiting_subdomains(subdomain_id,recv_dof_buffer[i+3+j]);
        }
      }
    }
    i += n_dofs+3;
  }

  // Sort the dofs associated to waiting processors (subdomains) and suppress
  // duplicates
  task.compress_waiting_subdomains();
}  

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::build_required_tasks_maps()
{  
  const unsigned int n_proc(comm->NumProc());
  const unsigned int n_tasks(tasks.size());
  MPI_Comm mpi_comm(comm->GetMpiComm());

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
          tasks[i].get_id());
      send_dof_buffer[current_offset+1] = static_cast<types::global_dof_index>(
          tasks[i].get_waiting_tasks_id(j));
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
void RadiativeTransfer<dim,tensor_dim>::build_local_required_tasks_map(Task &task,
    types::global_dof_index* recv_dof_buffer,int* recv_dof_disps_x,
    const unsigned int recv_dof_buffer_size)
{
  const unsigned int current_task_id(task.get_id());

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
    {
      for (unsigned int j=0; j<n_dofs; ++j)
      {
          std::pair<types::subdomain_id,unsigned int> subdomain_task_pair(
              subdomain_id,sender_task_id);
          task.add_to_required_tasks(subdomain_task_pair,recv_dof_buffer[i+3+j]);
      }
    }
    i += n_dofs+3;
  }
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::build_global_required_tasks()
{
  const unsigned int n_tasks(tasks.size());
  AssertIndexRange(0,n_tasks);
  const types::subdomain_id subdomain_id(tasks[0].get_subdomain_id());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>> tmp_map;
  // Loop over all the required_tasks and create a temporary map with sorted
  // dofs
  for (unsigned int i=0; i<n_tasks; ++i)
  {
    std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
      std::vector<types::global_dof_index>,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>>* required_tasks(
          tasks[i].get_required_tasks());
    std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
      std::vector<types::global_dof_index>,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>>::const_iterator 
        map_it(required_tasks->cbegin());
    std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
      std::vector<types::global_dof_index>,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>>::const_iterator 
        map_end(required_tasks->cend());
    for (; map_it!=map_end; ++map_it)
    {
      // Check that the required task is not owned by the processor
      if (std::get<0>(map_it->first)!=subdomain_id)
      {
        Assert(std::is_sorted(tmp_map[map_it->first].begin(),
              tmp_map[map_it->first].end()),ExcMessage(
              "The temporary map tmp_map is not sorted."));
        Assert(std::is_sorted(map_it->second.begin(),map_it->second.end()),
            ExcMessage("The vector of dofs of the required task is not sorted."));

        std::vector<types::global_dof_index> sorted_union(
            tmp_map[map_it->first].size()+map_it->second.size());
        std::vector<types::global_dof_index>::iterator vector_it;
        vector_it = std::set_union(tmp_map[map_it->first].begin(),
            tmp_map[map_it->first].end(),map_it->second.begin(),map_it->second.end(),
            sorted_union.begin());
        sorted_union.resize(vector_it-sorted_union.begin());
        tmp_map[map_it->first] = sorted_union;
      }
    }
  }

  // Build the global_required_tasks map, by looping over the tmp_map and
  // adding the position of the dofs in the MIP messages
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
      tmp_map_it(tmp_map.begin());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
      tmp_map_end(tmp_map.end());
  for (; tmp_map_it!=tmp_map_end; ++tmp_map_it)
  {
    const unsigned int i_max(tmp_map_it->second.size());
    for (unsigned int i=0; i<i_max; ++i)
      global_required_tasks[tmp_map_it->first][tmp_map_it->second[i]] = i;
  }
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::get_task_local_dof_indices(Task &task,
    std::vector<types::global_dof_index> &local_dof_indices)
{
  std::vector<unsigned int> const* sweep_order(task.get_sweep_order());
  const unsigned int sweep_order_size(task.get_sweep_order_size());
  // Loop over the cells in the sweep associated to this task
  for (unsigned int i=0; i<sweep_order_size; ++i)
  {
    // Copy the dofs associated to the cell to local_dof_indices
    typename DoFHandler<dim>::active_cell_iterator cell(
        fecell_mesh[(*sweep_order)[i]].get_cell());
    std::vector<types::global_dof_index> cell_dof_indices(tensor_dim);
    cell->get_dof_indices(cell_dof_indices);
    for (unsigned int j=0; j<tensor_dim; ++j)
      local_dof_indices[i*tensor_dim+j] = cell_dof_indices[j];
  }
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::initialize_scheduler() const
{
  n_tasks_to_execute = tasks.size();
  // Add to the tasks_ready list all the tasks that do not require another
  // task to start
  for (unsigned int i=0; i<n_tasks_to_execute; ++i)
    if (tasks[i].get_n_required_tasks()==0)
      tasks_ready.push_back(i);
}

template <int dim,int tensor_dim>
Task const* const RadiativeTransfer<dim,tensor_dim>::get_next_task() const
{
  // If tasks_ready is empty, we need to wait to receive data
  while (tasks_ready.size()==0)
  {
    receive_angular_flux();
  }

  // Pop a task from the tasks_ready list and decrease the number of tasks
  // that are left to execute by
  --n_tasks_to_execute;
  unsigned int i(tasks_ready.front());
  tasks_ready.pop_front();
  return &tasks[i];
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::send_angular_flux(Task const &task,
    std::list<double*> &buffers,std::list<MPI_Request*> &requests,
    std::unordered_map<types::global_dof_index,double> &angular_flux) const
{
  MPI_Comm mpi_comm(comm->GetMpiComm());
  const unsigned int n_tasks(tasks.size());

  std::unordered_map<types::subdomain_id,std::vector<types::global_dof_index>>*
    const waiting_subdomains(task.get_waiting_subdomains());
  std::unordered_map<types::subdomain_id,std::vector<types::global_dof_index>>
    ::iterator map_it(waiting_subdomains->begin());
  std::unordered_map<types::subdomain_id,std::vector<types::global_dof_index>>
    ::iterator map_end(waiting_subdomains->end());

  // Loop over all the waiting processors
  types::subdomain_id source(task.get_subdomain_id());
  unsigned int tag(task.get_id());
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
        buffer[i] = angular_flux[map_it->second[i]];
      buffers.push_back(buffer);
      MPI_Request* request = new MPI_Request;
      requests.push_back(request);

      MPI_Isend(buffer,count,MPI_DOUBLE,destination,tag,mpi_comm,request);
    }                                               
    else
    {
      // If source and destination processors are the same, loop over the
      // tasks and set the required dofs if necessary
      std::pair<types::subdomain_id,unsigned int> current_task(source,tag);
      for (unsigned int i=0; i<n_tasks; ++i)
      {
        if (tasks[i].is_task_required(current_task)==true)
        {
          std::vector<types::global_dof_index> const* const required_dofs(
              tasks[i].get_required_dofs(current_task));
          const unsigned int required_dofs_size(required_dofs->size());
          for (unsigned int j=0; j<required_dofs_size; ++j)
            tasks[i].set_required_dof((*required_dofs)[j],
                angular_flux[(*required_dofs)[j]]);
        }

        // If all the required dofs are known, i.e., the task is ready, the
        // tasks is added to the tasks_ready list
        if (tasks[i].is_ready()==true)
          tasks_ready.push_back(i);
      }
    }
  }
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::receive_angular_flux() const 
{
  MPI_Comm mpi_comm(comm->GetMpiComm());
  const unsigned int n_tasks(tasks.size());

  // Loop on the global_required_tasks map
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::unordered_map<types::global_dof_index,unsigned int>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
      global_map_it(global_required_tasks.begin());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::unordered_map<types::global_dof_index,unsigned int>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
      global_map_end(global_required_tasks.end());
  for (; global_map_it!=global_map_end; ++global_map_it)
  {
    types::subdomain_id source(std::get<0>(global_map_it->first));
    unsigned int tag(std::get<1>(global_map_it->first));
    int flag;

    // Check if we can receive a message
    MPI_Iprobe(source,tag,mpi_comm,&flag,MPI_STATUS_IGNORE);

    if (flag==true)
    {
      int count(global_map_it->second.size());
      double* buffer = new double [count];

      // Receive the message
      MPI_Recv(buffer,count,MPI_DOUBLE,source,tag,mpi_comm,MPI_STATUS_IGNORE);

      // Set the required dofs
      for (unsigned int i=0; i<n_tasks; ++i)
      {
        if (tasks[i].is_task_required(global_map_it->first)==true)
        { 
          std::vector<types::global_dof_index> const* const required_dofs(
              tasks[i].get_required_dofs(global_map_it->first));
          const unsigned int required_dofs_size(required_dofs->size());
          for (unsigned int j=0; j<required_dofs_size; ++j)
          {
            const unsigned int required_dof((*required_dofs)[j]);
            const unsigned int buffer_pos(global_map_it->second[required_dof]);
            tasks[i].set_required_dof(required_dof,buffer[buffer_pos]);
          }
        }

        // If the task has all the required dofs, it goes into the tasks_ready
        // list
        if (tasks[i].is_ready()==true)
          tasks_ready.push_back(i);
      }
      delete [] buffer;
    }
  }
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::free_buffers(
    std::list<double*> &buffers,std::list<MPI_Request*> &requests) const
{  
  std::list<double*>::iterator buffers_it(buffers.begin());
  std::list<double*>::iterator buffers_end(buffers.end());
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
  // Initialize the scheduler by creating the tasks_ready list
  initialize_scheduler();
  // Sweep through the mesh
  while (n_tasks_to_execute!=0)
  {
    sweep(*get_next_task(),buffers,requests,y);
    free_buffers(buffers,requests);
  }

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
    for (unsigned int i=0; i<tensor_dim; ++i)
      x_cell[i] = x[0][local_dof_indices[i]];
    
    Tensor<1,tensor_dim> scat_src_cell((*(fecell->get_mass_matrix()))*x_cell);
    for (unsigned int j=0; j<n_mom; ++j)
    {
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
  std::unordered_map<types::global_dof_index,double> angular_flux;
 
  // Sweep on the spatial cells of the current task
  for (unsigned int i=0; i<sweep_order_size; ++i)
  {
    FECell<dim,tensor_dim> const* const fecell = &fecell_mesh[(*sweep_order)[i]];
    typename DoFHandler<dim>::active_cell_iterator const cell(fecell->get_cell());
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
      // Divide the source by the sum of the weights to the input source is
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
      Point<dim> const* const normal_vector = fecell->get_normal_vector(face);
      double n_dot_omega(0.);
      for (unsigned int d=0; d<dim; ++d)
        n_dot_omega += (*omega)[d]*(*normal_vector)(d);

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
          typename DoFHandler<dim>::active_cell_iterator neighbor_cell;
          neighbor_cell = cell->neighbor(face);
          std::vector<types::global_dof_index> neighbor_dof_indices(tensor_dim);
          neighbor_cell->get_dof_indices(neighbor_dof_indices);
          // This assumes that there is only one cell per task. If there are
          // more than one cell per task, a local angular flux vector is
          // needed.
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
                  (quad->is_most_normal_direction(idir)==true))||
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
        flux_moments[mom][multivector_indices[j]] += d2m*x[j];
    }

    // Send the angular flux to the others processors
    for (unsigned int j=0; j<tensor_dim; ++j)
#ifdef DEAL_II_USE_LARGE_INDEX_TYPE
      angular_flux[map->GID64(multivector_indices[j])] = x[j];
#else
      angular_flux[map->GID(multivector_indices[j])] = x[j];
#endif 
  }

  // Delete all required_dofs map that is now useless
  task.clear_required_dofs();

  // Send angular_flux to the waiting task
  send_angular_flux(task,buffers,requests,angular_flux);
}

template <int dim,int tensor_dim>
void RadiativeTransfer<dim,tensor_dim>::get_multivector_indices(
    std::vector<int> &dof_indices,
    typename DoFHandler<dim>::active_cell_iterator const& cell) const
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
