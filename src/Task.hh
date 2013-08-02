/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _TASK_HH_
#define _TASK_HH_

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>
#include "boost/functional/hash/hash.hpp"
#include "deal.II/base/exceptions.h"
#include "deal.II/base/types.h"

using namespace dealii;


class Task
{
  public :
    Task(unsigned int idir,unsigned int id,types::subdomain_id subdomain_id,
        std::vector<unsigned int> &sweep_order,
        std::vector<std::pair<types::subdomain_id,
        std::vector<types::global_dof_index>>> &incomplete_required_tasks);

    void add_to_required_tasks(std::pair<types::subdomain_id,unsigned int> 
        &subdomain_task_pair,types::global_dof_index dof);

    void add_to_waiting_tasks(std::pair<types::subdomain_id,
        unsigned int> &subdomain_task_pair,types::global_dof_index dof);

    void add_to_waiting_subdomains(types::subdomain_id subdomain_id,
        types::global_dof_index dof);

    void compress_waiting_subdomains();

    bool is_ready() const;

    bool is_task_required(std::pair<types::subdomain_id,unsigned int> const 
        &current_task) const;

    unsigned int get_id() const;

    unsigned int get_idir() const;

    unsigned int get_sweep_order_size() const;

    unsigned int get_incomplete_n_required_tasks() const;

    unsigned int get_incomplete_subdomain_id(unsigned int i) const;

    unsigned int get_incomplete_n_dofs(unsigned int i) const;

    unsigned int get_n_waiting_tasks() const;

    unsigned int get_n_required_tasks() const;

    types::subdomain_id get_subdomain_id() const;

    // Rajouter le const
    unsigned int get_waiting_tasks_subdomain_id(unsigned int i);
    unsigned int get_waiting_tasks_id(unsigned int i) const;
    unsigned int get_waiting_tasks_n_dofs(unsigned int i) const;
    std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
      std::vector<types::global_dof_index>,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>>* get_required_tasks();
    std::vector<types::global_dof_index> const* get_waiting_tasks_dofs(
        unsigned int i);
    std::unordered_map<types::subdomain_id,std::vector<types::global_dof_index>>*
      const get_waiting_subdomains() const;
    void print();
    void clear_temporary_data();
    std::vector<types::global_dof_index>* const get_required_dofs(
        std::pair<types::global_dof_index,unsigned int> const &current_task) const;
    void set_ghost_dof(types::global_dof_index dof,double value) const;
    double get_ghost_angular_flux(types::global_dof_index) const;

    std::vector<unsigned int> const* get_sweep_order() const;

    std::vector<types::global_dof_index> const* get_incomplete_dofs(unsigned int i) 
      const;

  private :
    unsigned int idir;
    unsigned int weight;
    // Number not unique globally only per processor
    unsigned int id;
    unsigned int n_ghost_dofs;
    mutable unsigned int n_missing_dofs;
    types::subdomain_id subdomain_id;
    std::vector<unsigned int> sweep_order;
    std::vector<std::pair<types::subdomain_id,std::vector<types::global_dof_index>>> 
      incomplete_required_tasks;
    // Pair (subdomain_id, task_id (unsigned int)), dof of the flux passed
    // Required tasks before we can start this task
    mutable std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
      std::vector<types::global_dof_index>,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>> required_tasks;
    // Tasks waiting for this task to be done, needed for required_tasks
    std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
      std::vector<types::global_dof_index>,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>> waiting_tasks;
    // Subdomains waiting angular fluxes from this task, dofs have to be
    // sorted
    std::unordered_map<types::subdomain_id,std::vector<types::global_dof_index>>
      waiting_subdomains;

    // Need a pointer because of Trilinos
    mutable std::unordered_map<types::global_dof_index,double> ghost_dofs;
};

inline void Task::add_to_required_tasks(std::pair<types::subdomain_id,unsigned int>
    &subdomain_task_pair,types::global_dof_index dof)
{
  required_tasks[subdomain_task_pair].push_back(dof);
  ++n_ghost_dofs;
}

inline void Task::add_to_waiting_tasks(std::pair<types::subdomain_id,unsigned int> 
    &subdomain_task_pair,types::global_dof_index dof)
{
  waiting_tasks[subdomain_task_pair].push_back(dof);
}

inline void Task::add_to_waiting_subdomains(types::subdomain_id,
    types::global_dof_index dof)
{
  waiting_subdomains[subdomain_id].push_back(dof);
}
    
inline bool Task::is_task_required(std::pair<types::subdomain_id,unsigned int> 
    const &current_task) const
{
  return required_tasks.count(current_task);
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
  AssertIndexRange(i,incomplete_required_tasks.size());
  return std::get<0>(incomplete_required_tasks[i]);
}

inline unsigned int Task::get_incomplete_n_dofs(unsigned int i) const
{
  AssertIndexRange(i,incomplete_required_tasks.size());
  return std::get<1>(incomplete_required_tasks[i]).size();
}

inline unsigned int Task::get_n_waiting_tasks() const
{
  return waiting_tasks.size();
}

inline unsigned int Task::get_n_required_tasks() const
{
  return required_tasks.size();
}

inline types::subdomain_id Task::get_waiting_tasks_subdomain_id(unsigned int i)
{
  AssertIndexRange(i,waiting_tasks.size());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::const_iterator map_it(
        waiting_tasks.cbegin());

  for (unsigned int j=0; j<i; ++j,++map_it);

  return std::get<0>(map_it->first);
}

inline unsigned int Task::get_waiting_tasks_id(unsigned int i) const
{
  AssertIndexRange(i,waiting_tasks.size());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::const_iterator map_it(
        waiting_tasks.cbegin());

  for (unsigned int j=0; j<i; ++j,++map_it);

  return std::get<1>(map_it->first);
}

inline unsigned int Task::get_waiting_tasks_n_dofs(unsigned int i) const
{
  AssertIndexRange(i,waiting_tasks.size());
  std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
    std::vector<types::global_dof_index>,
    boost::hash<std::pair<types::subdomain_id,unsigned int>>>::const_iterator map_it(
        waiting_tasks.cbegin());

  for (unsigned int j=0; j<i; ++j,++map_it);

  return map_it->second.size();
}

inline double Task::get_ghost_angular_flux(types::global_dof_index dof) const
{
  return ghost_dofs[dof];
}

inline std::vector<types::global_dof_index> const* Task::get_waiting_tasks_dofs(
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
 
inline types::subdomain_id Task::get_subdomain_id() const
{
  return subdomain_id;
}

inline std::unordered_map<types::subdomain_id,std::vector<types::global_dof_index>>*
  const Task::get_waiting_subdomains() const
{
  return const_cast<std::unordered_map<types::subdomain_id,
         std::vector<types::global_dof_index>>*> (&waiting_subdomains);
}
    
inline std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
      std::vector<types::global_dof_index>,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>>* 
Task::get_required_tasks()
{
  return &required_tasks;
}

inline std::vector<unsigned int> const* Task::get_sweep_order() const
{
  return &sweep_order;
}

inline std::vector<types::global_dof_index> const* Task::get_incomplete_dofs(
    unsigned int i) const
{
  AssertIndexRange(i,incomplete_required_tasks.size());
  return &std::get<1>(incomplete_required_tasks[i]);
}

inline std::vector<types::global_dof_index>* const 
Task::get_required_dofs(std::pair<types::global_dof_index,unsigned int> 
    const &current_task) const
{   
  return const_cast<std::vector<types::global_dof_index>* const> (
      &(required_tasks[current_task]));
}

inline void Task::clear_temporary_data()
{
  incomplete_required_tasks.clear();
  waiting_tasks.clear();
}

#endif
