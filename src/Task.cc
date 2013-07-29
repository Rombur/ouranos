/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "Task.hh"

Task::Task(unsigned int idir,unsigned int id,std::vector<unsigned int> &sweep_order,
    std::vector<std::pair<types::subdomain_id,
    std::vector<types::global_dof_index>>> &incomplete_required_tasks) :
  done(false),
  idir(idir),
  id(id),
  sweep_order(sweep_order),
  incomplete_required_tasks(incomplete_required_tasks)
{}

void Task::print()
{
  std::cout<<"ID "<<id<<std::endl;
  std::cout<<"idir "<<idir<<std::endl;
  std::cout<<"sweep order"<<std::endl;
  for (unsigned int i=0; i<sweep_order.size(); ++i)
    std::cout<<sweep_order[i]<<std::endl;
  std::cout<<"waiting tasks"<<std::endl;
  std::unordered_map<types::global_dof_index,std::vector<std::pair<types::subdomain_id,
    unsigned int>>>::iterator waiting_map_it(waiting_tasks.begin());
  std::unordered_map<types::global_dof_index,std::vector<std::pair<types::subdomain_id,
    unsigned int>>>::iterator waiting_map_end(waiting_tasks.end());
  for (; waiting_map_it!=waiting_map_end; ++waiting_map_it)
  {
    for (unsigned int i=0; i<waiting_map_it->second.size(); ++i)
      std::cout<<waiting_map_it->first<<" "<<std::get<0>(waiting_map_it->second[i])
        <<" "<<std::get<1>(waiting_map_it->second[i])<<std::endl;
  }
  std::cout<<"required tasks"<<std::endl;
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>>::iterator required_map_it(
        required_tasks.begin());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>>::iterator required_map_end(
        required_tasks.end());
  for (; required_map_it!=required_map_end; ++required_map_it)
  {
    std::cout<<std::get<0>(required_map_it->first)<<" "<<
      std::get<1>(required_map_it->first)<<" ";
    std::vector<types::global_dof_index> dofs(required_map_it->second);
    for (unsigned int i=0; i<dofs.size(); ++i)
      std::cout<<dofs[i]<<" ";
    std::cout<<std::endl;
  }
  std::cout<<std::endl;
}
