/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _TASK_HH_
#define _TASK_HH_

#include <iostream>
#include <map>
#include <utility>
#include <vector>
#include "deal.II/base/types.h"

using namespace dealii;


class Task
{
  public :
    Task(unsigned int idir,unsigned int id,std::vector<unsigned int> &sweep_order,
        std::vector<std::pair<types::subdomain_id,
        std::vector<types::global_dof_index>>> &incomplete_required_tasks);

    void add_to_required_tasks(std::pair<types::subdomain_id,unsigned int> 
        &subdomain_task_pair,types::global_dof_index dof);

    void add_to_waiting_tasks(types::global_dof_index dof,
        std::pair<types::subdomain_id,unsigned int> &subdomain_task_pair);

    unsigned int get_id() const;

    unsigned int get_idir() const;

    unsigned int get_sweep_order_size() const;

    unsigned int get_incomplete_n_required_tasks() const;

    unsigned int get_incomplete_subdomain_id(unsigned int i) const;

    unsigned int get_incomplete_n_dofs(unsigned int i) const;

    unsigned int get_waiting_tasks_size() const;

    // Rajouter le const
    std::vector<unsigned int> get_waiting_subdomain_id(unsigned int i);
    std::vector<std::pair<types::subdomain_id,unsigned int>> get_waiting_tasks(unsigned int i);
    std::map<types::global_dof_index,std::vector<std::pair<types::subdomain_id,unsigned int>>>
    get_waiting_tasks_map();
    void print();

    std::vector<unsigned int> const* get_sweep_order() const;

    std::vector<types::global_dof_index> const* get_incomplete_dofs(unsigned int i) 
      const;

  private :
    bool done;
    unsigned int idir;
    unsigned int weight;
    // Number not unique globally only per processor
    unsigned int id;
    std::vector<unsigned int> sweep_order;
    std::vector<std::pair<types::subdomain_id,std::vector<types::global_dof_index>>> 
      incomplete_required_tasks;
    // Pair (subdomain_id, task_id (unsigned int)), dof of the flux passed
    // Required tasks before we can start this task
    std::map<std::pair<types::subdomain_id,unsigned int>,
      std::vector<types::global_dof_index>> required_tasks;
    // Tasks waiting for this task to be done
    std::map<types::global_dof_index,std::vector<std::pair<types::subdomain_id,
      unsigned int>>> waiting_tasks;
};

inline void Task::add_to_required_tasks(std::pair<types::subdomain_id,unsigned int>
    &subdomain_task_pair,types::global_dof_index dof)
{
  required_tasks[subdomain_task_pair].push_back(dof);
}

inline void Task::add_to_waiting_tasks(types::global_dof_index dof,
    std::pair<types::subdomain_id,unsigned int> &subdomain_task_pair)
{
  if (waiting_tasks.count(dof)==0)
    waiting_tasks.insert(std::pair<types::global_dof_index,
        std::vector<std::pair<types::subdomain_id,unsigned int>>>
        (dof,std::vector<std::pair<types::subdomain_id,unsigned int>> 
         (1,subdomain_task_pair)));
  else
    waiting_tasks[dof].push_back(subdomain_task_pair);
}

inline unsigned int Task::get_id() const
{
  return id;
}

inline unsigned int Task::get_idir() const
{
  return idir;
}
    
inline unsigned int Task::get_sweep_order_size() const
{
  return sweep_order.size();
}

inline unsigned int Task::get_incomplete_n_required_tasks() const
{
  return incomplete_required_tasks.size();
}

inline unsigned int Task::get_incomplete_subdomain_id(unsigned int i) const
{
  return std::get<0>(incomplete_required_tasks[i]);
}

inline unsigned int Task::get_incomplete_n_dofs(unsigned int i) const
{
  return std::get<1>(incomplete_required_tasks[i]).size();
}

inline unsigned int Task::get_waiting_tasks_size() const
{
  return waiting_tasks.size();
}

inline std::vector<unsigned int> Task::get_waiting_subdomain_id(unsigned int i) 
{
  std::vector<unsigned int> vector;
  std::map<types::global_dof_index,std::vector<std::pair<types::subdomain_id,
    unsigned int>>>::iterator map_it(waiting_tasks.begin());

  for (unsigned int j=0; j<i; ++j,++map_it);

  for (unsigned int j=0; j<map_it->second.size(); ++j)
    vector.push_back(std::get<0>(map_it->second[j]));

  return vector;
}

inline std::map<types::global_dof_index,std::vector<std::pair<types::subdomain_id,unsigned int>>>
    Task::get_waiting_tasks_map()
{
  return waiting_tasks;
}
    
inline std::vector<std::pair<types::subdomain_id,unsigned int>> Task::get_waiting_tasks(
    unsigned int i)
{
  std::map<types::global_dof_index,std::vector<std::pair<types::subdomain_id,
    unsigned int>>>::iterator map_it(waiting_tasks.begin());

  for (unsigned int j=0; j<i; ++j,++map_it);
  
  return map_it->second;
}

inline std::vector<unsigned int> const* Task::get_sweep_order() const
{
  return &sweep_order;
}

inline std::vector<types::global_dof_index> const* Task::get_incomplete_dofs(
    unsigned int i) const
{
  return &std::get<1>(incomplete_required_tasks[i]);
}

#endif
