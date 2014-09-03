/* Copyright (c) 2013, 2014 Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "Task.hh"

#include <algorithm>

Task::Task(unsigned int idir,unsigned int id,types::subdomain_id subdomain_id,
    std::vector<unsigned int> &sweep_order,std::vector<std::pair<types::subdomain_id,
    std::vector<types::global_dof_index>>> &incomplete_required_tasks) :
  idir(idir),
  id(id),
  n_required_dofs(0),
  subdomain_id(subdomain_id),
  sweep_order(sweep_order),
  incomplete_required_tasks(incomplete_required_tasks)
{}

void Task::compress_waiting_tasks()
{
  std::vector<task_tuple> tmp;
  // Group the elements of the vector that have the same subdomain_id.
  for (auto &wait_task : waiting_tasks)
  {
    const types::subdomain_id sub_id(std::get<0>(wait_task));
    const unsigned int task_id(std::get<1>(wait_task));
    std::vector<task_tuple>::iterator tmp_it(tmp.begin());
    std::vector<task_tuple>::iterator tmp_end(tmp.end());
    for (; tmp_it!=tmp_end; ++tmp_it)
      if ((std::get<0>(*tmp_it)==sub_id) && (std::get<1>(*tmp_it)==task_id))
        break;

    if (tmp_it!=tmp_end)
      std::get<2>(*tmp_it).insert(std::get<2>(*tmp_it).end(),
          std::get<2>(wait_task).begin(),std::get<2>(wait_task).end());
    else
      tmp.push_back(wait_task);
  }

  waiting_tasks = tmp;
  std::vector<types::global_dof_index>::iterator vector_end;
  for (auto &wait_task : waiting_tasks)
  {
    // Sort the dofs.
    std::sort(std::get<2>(wait_task).begin(),std::get<2>(wait_task).end());
    // Delete repeated values.
    vector_end = std::unique(std::get<2>(wait_task).begin(),std::get<2>(wait_task).end());
    std::get<2>(wait_task).resize(std::distance(std::get<2>(wait_task).begin(),vector_end));
  }
}

void Task::compress_waiting_subdomains()
{
  std::vector<domain_pair> tmp;
  // Group the elements of the vector that have the same subdomain_id
  for (auto &wait_subdom : waiting_subdomains)
  {
    const types::subdomain_id sub_id(wait_subdom.first);
    std::vector<domain_pair>::iterator tmp_it(tmp.begin());
    std::vector<domain_pair>::iterator tmp_end(tmp.end());
    for (; tmp_it!=tmp_end; ++tmp_it)
      if (tmp_it->first==sub_id)
        break;
    
    if (tmp_it!=tmp_end)
      tmp_it->second.insert(tmp_it->second.end(),wait_subdom.second.begin(),
          wait_subdom.second.end());
    else
      tmp.push_back(wait_subdom);
  }

  waiting_subdomains = tmp;
  std::vector<types::global_dof_index>::iterator vector_end;
  for (auto &wait_subdom : waiting_subdomains)
  {
    // Sort the dofs
    std::sort(wait_subdom.second.begin(),wait_subdom.second.end());
    // Delete repeated values
    vector_end = std::unique(wait_subdom.second.begin(),wait_subdom.second.end());
    wait_subdom.second.resize(std::distance(wait_subdom.second.begin(),vector_end));
  }
}

void Task::add_to_required_tasks(types::subdomain_id s_id,unsigned int t_id,types::global_dof_index* recv,
    unsigned int pos,unsigned int n)
{
  std::vector<types::global_dof_index> tmp(&recv[pos],&recv[pos+n]);
  required_tasks.push_back(task_tuple(s_id,t_id,tmp));
  n_required_dofs += n;
  n_missing_dofs = n_required_dofs;
}

bool Task::is_ready() const
{
  bool ready(false);
  if (n_missing_dofs==0)
  {
    ready = true;
    n_missing_dofs = n_required_dofs;
  }

  return ready;
}

void Task::clear_temporary_data()
{
  incomplete_required_tasks.clear();
  waiting_tasks.clear();
}

void Task::finalize_maps()
{
  // Create local_waiting_tasks
  const unsigned int waiting_tasks_size(waiting_tasks.size());
  for (unsigned int i=0; i<waiting_tasks_size; ++i)
  {
    if (std::get<0>(waiting_tasks[i])==subdomain_id)
    {
      local_waiting_tasks.push_back(std::pair<unsigned int,std::vector<types::global_dof_index>> 
          (std::get<1> (waiting_tasks[i]),std::get<2>(waiting_tasks[i])));
    }
  }

  // Create required_tasks_map
  const unsigned int required_tasks_size(required_tasks.size());
  for (unsigned int i=0; i<required_tasks_size; ++i)
  {
    std::pair<types::subdomain_id,unsigned int> pair(std::get<0>(required_tasks[i]),
        std::get<1>(required_tasks[i]));
    required_tasks_map[pair] = i;
  }

  // Clear incomplete_required_tasks and waiting_tasks
  clear_temporary_data();
}

void Task::print(std::ostream &output_stream)
{
  output_stream<<"ID "<<id<<std::endl;
  output_stream<<"idir "<<idir<<std::endl;
  output_stream<<"sweep order"<<std::endl;
  for (unsigned int i=0; i<sweep_order.size(); ++i)
    output_stream<<sweep_order[i]<<std::endl;
  output_stream<<"waiting tasks "<<waiting_tasks.size()<<std::endl;
  for (unsigned int i=0; i<waiting_tasks.size(); ++i)
  {
    output_stream<<std::get<0>(waiting_tasks[i])<<" "<<
      std::get<1>(waiting_tasks[i])<<" ";
    std::vector<types::global_dof_index> dofs(std::get<2>(waiting_tasks[i]));
    for (unsigned int i=0; i<dofs.size(); ++i)
      output_stream<<dofs[i]<<" ";
    output_stream<<std::endl;
  }
  output_stream<<"required tasks "<<required_tasks.size()<<std::endl;
  for (unsigned int i=0; i<required_tasks.size(); ++i)
  {
    output_stream<<std::get<0>(required_tasks[i])<<" "<<
      std::get<1>(required_tasks[i])<<" ";
    std::vector<types::global_dof_index> dofs(std::get<2>(required_tasks[i]));
    for (unsigned int i=0; i<dofs.size(); ++i)
      output_stream<<dofs[i]<<" ";
    output_stream<<std::endl;
  }
  output_stream<<std::endl;
}
