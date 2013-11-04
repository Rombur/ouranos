/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "Task.hh"

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

void Task::compress_waiting_subdomains()
{
  std::unordered_map<types::subdomain_id,std::vector<types::global_dof_index>>
    ::iterator map_it(waiting_subdomains.begin());
  std::unordered_map<types::subdomain_id,std::vector<types::global_dof_index>>
    ::iterator map_end(waiting_subdomains.end());
  std::vector<types::global_dof_index>::iterator vector_end;
  for (; map_it!=map_end; ++map_it)
  {
    // Sort the dofs
    std::sort(map_it->second.begin(),map_it->second.end());
    // Delete repeated values
    vector_end = std::unique(map_it->second.begin(),map_it->second.end());
    map_it->second.resize(std::distance(map_it->second.begin(),vector_end));
  }
}

void Task::add_to_required_tasks(std::pair<types::subdomain_id,unsigned int>
    &subdomain_task_pair,types::global_dof_index dof)
{
  required_tasks[subdomain_task_pair].push_back(dof);
  ++n_required_dofs;
  n_missing_dofs = n_required_dofs;
}

void Task::set_required_dof(types::global_dof_index dof,double value) const
{
  required_dofs[dof] = value;
  --n_missing_dofs;
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

types::subdomain_id Task::get_waiting_tasks_subdomain_id(unsigned int i) const
{
  AssertIndexRange(i,waiting_tasks.size());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::const_iterator map_it(
        waiting_tasks.cbegin());

  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::const_iterator p_it(
        waiting_tasks.cend());

  for (unsigned int j=0; j<i; ++j,++map_it);

  return std::get<0>(map_it->first);
}

unsigned int Task::get_waiting_tasks_n_dofs(unsigned int i) const
{
  AssertIndexRange(i,waiting_tasks.size());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::const_iterator map_it(
        waiting_tasks.cbegin());

  for (unsigned int j=0; j<i; ++j,++map_it);

  return map_it->second.size();
}

unsigned int Task::get_waiting_tasks_id(unsigned int i) const
{
  AssertIndexRange(i,waiting_tasks.size());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::const_iterator map_it(
        waiting_tasks.cbegin());

  for (unsigned int j=0; j<i; ++j,++map_it);

  return std::get<1>(map_it->first);
}

std::vector<types::global_dof_index> const* Task::get_waiting_tasks_dofs(
    unsigned int i)
{
  AssertIndexRange(i,waiting_tasks.size());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::const_iterator map_it(
        waiting_tasks.cbegin());

  for (unsigned int j=0; j<i; ++j,++map_it);

  return &(map_it->second);
}

void Task::print(std::ostream &output_stream)
{
  output_stream<<"ID "<<id<<std::endl;
  output_stream<<"idir "<<idir<<std::endl;
  output_stream<<"sweep order"<<std::endl;
  for (unsigned int i=0; i<sweep_order.size(); ++i)
    output_stream<<sweep_order[i]<<std::endl;
  output_stream<<"waiting tasks "<<waiting_tasks.size()<<std::endl;
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
      waiting_map_it(waiting_tasks.begin());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
      waiting_map_end(waiting_tasks.end());
  for (; waiting_map_it!=waiting_map_end; ++waiting_map_it)
  {
    output_stream<<std::get<0>(waiting_map_it->first)<<" "<<
      std::get<1>(waiting_map_it->first)<<" ";
    std::vector<types::global_dof_index> dofs(waiting_map_it->second);
    for (unsigned int i=0; i<dofs.size(); ++i)
      output_stream<<dofs[i]<<" ";
    output_stream<<std::endl;
  }
  output_stream<<"required tasks"<<std::endl;
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
      required_map_it(required_tasks.begin());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
      required_map_end(required_tasks.end());
  for (; required_map_it!=required_map_end; ++required_map_it)
  {
    output_stream<<std::get<0>(required_map_it->first)<<" "<<
      std::get<1>(required_map_it->first)<<" ";
    std::vector<types::global_dof_index> dofs(required_map_it->second);
    for (unsigned int i=0; i<dofs.size(); ++i)
      output_stream<<dofs[i]<<" ";
    output_stream<<std::endl;
  }
  output_stream<<std::endl;
}
