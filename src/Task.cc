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
  std::vector<domain_pair> tmp;
  std::vector<domain_pair>::iterator waiting_it(waiting_subdomains.begin());
  std::vector<domain_pair>::iterator waiting_end(waiting_subdomains.end());
  // Group the elements of the vector that have the same subdomain_id
  for (; waiting_it!=waiting_end; ++waiting_it)
  {
    const unsigned int sub_id(waiting_it->first);
    std::vector<domain_pair>::iterator tmp_it(tmp.begin());
    std::vector<domain_pair>::iterator tmp_end(tmp.end());
    for (; tmp_it!=tmp_end; ++tmp_it)
      if (tmp_it->first==sub_id)
        break;
    
    if (tmp_it!=tmp_end)
      tmp_it->second.insert(tmp_it->second.end(),waiting_it->second.begin(),waiting_it->second.end());
    else
      tmp.push_back(*waiting_it);
  }

  waiting_subdomains = tmp;
  waiting_it = waiting_subdomains.begin();
  waiting_end = waiting_subdomains.end();
  std::vector<types::global_dof_index>::iterator vector_end;
  for (; waiting_it!=waiting_end; ++waiting_it)
  {
    // Sort the dofs
    std::sort(waiting_it->second.begin(),waiting_it->second.end());
    //Delete repeated values
    vector_end = std::unique(waiting_it->second.begin(),waiting_it->second.end());
    waiting_it->second.resize(std::distance(waiting_it->second.begin(),vector_end));
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
//  output_stream<<"ID "<<id<<std::endl;
//  output_stream<<"idir "<<idir<<std::endl;
//  output_stream<<"sweep order"<<std::endl;
//  for (unsigned int i=0; i<sweep_order.size(); ++i)
//    output_stream<<sweep_order[i]<<std::endl;
//  output_stream<<"waiting tasks "<<waiting_tasks.size()<<std::endl;
//  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
//    std::vector<types::global_dof_index>,
//    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
//      waiting_map_it(waiting_tasks.begin());
//  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
//    std::vector<types::global_dof_index>,
//    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
//      waiting_map_end(waiting_tasks.end());
//  for (; waiting_map_it!=waiting_map_end; ++waiting_map_it)
//  {
//    output_stream<<std::get<0>(waiting_map_it->first)<<" "<<
//      std::get<1>(waiting_map_it->first)<<" ";
//    std::vector<types::global_dof_index> dofs(waiting_map_it->second);
//    for (unsigned int i=0; i<dofs.size(); ++i)
//      output_stream<<dofs[i]<<" ";
//    output_stream<<std::endl;
//  }
//  output_stream<<"required tasks"<<std::endl;
//  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
//    std::vector<types::global_dof_index>,
//    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
//      required_map_it(required_tasks.begin());
//  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
//    std::vector<types::global_dof_index>,
//    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::iterator 
//      required_map_end(required_tasks.end());
//  for (; required_map_it!=required_map_end; ++required_map_it)
//  {
//    output_stream<<std::get<0>(required_map_it->first)<<" "<<
//      std::get<1>(required_map_it->first)<<" ";
//    std::vector<types::global_dof_index> dofs(required_map_it->second);
//    for (unsigned int i=0; i<dofs.size(); ++i)
//      output_stream<<dofs[i]<<" ";
//    output_stream<<std::endl;
//  }
//  output_stream<<std::endl;
}
