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

    /// Add a dof index to the given required task.
    void add_to_required_tasks(std::pair<types::subdomain_id,unsigned int> 
        &subdomain_task_pair,types::global_dof_index dof);

    /// Add a dof index to the given waiting task.
    void add_to_waiting_tasks(std::pair<types::subdomain_id,
        unsigned int> &subdomain_task_pair,types::global_dof_index dof);

    /// Add a dof index to the given waiting subdomain.
    void add_to_waiting_subdomains(types::subdomain_id other_subdomain_id,
        types::global_dof_index dof);

    /// Clear the required_dofs map.
    void clear_required_dofs() const;

    /// Clear incomplete_required_tasks and waiting_tasks.
    void clear_temporary_data();

    /// Sort the dofs associated to waiting processors (subdomains) and
    /// suppress duplicates.
    void compress_waiting_subdomains();

    /// Output the data for debug purpose.
    void print(std::ostream &output_stream);

    /// Set the given angular flux value to the given dof index. 
    void set_required_dof(types::global_dof_index dof,double value) const;

    /// Return true if the task is ready for sweep.
    bool is_ready() const;

    /// Return true is the given task is required by the current task.
    bool is_task_required(std::pair<types::subdomain_id,unsigned int> const 
        &current_task) const;

    /// Return the task ID.
    unsigned int get_id() const;

    /// Return the direction associated with the task.
    unsigned int get_idir() const;

    /// Return the number of dofs of the ith element of
    /// incomplete_required_tasks.
    unsigned int get_incomplete_n_dofs(unsigned int i) const;

    /// Return the number of required tasks in in incomplete_required_tasks,
    /// i.e, the size of incomplete_required_tasks.
    unsigned int get_incomplete_n_required_tasks() const;

    /// Return the subdomain ID oth the ith element of
    /// incomplete_required_tasks.
    unsigned int get_incomplete_subdomain_id(unsigned int i) const;

    /// Return the number of tasks required before the current task is ready.
    unsigned int get_n_required_tasks() const;

    /// Return the number of tasks that are waiting for the current task to
    /// finish.
    unsigned int get_n_waiting_tasks() const;
  
    /// Return the size of sweep_order, i.e., the number of cells in the task.
    unsigned int get_sweep_order_size() const;

    /// Return the ID of the ith element in waiting_tasks.
    unsigned int get_waiting_tasks_id(unsigned int i) const;
    
    /// Return the number of dofs of the ith element in waiting_tasks.
    unsigned int get_waiting_tasks_n_dofs(unsigned int i) const;

    /// Return the subdomain ID of the ith element in waiting_tasks.
    types::subdomain_id get_waiting_tasks_subdomain_id(unsigned int i) const;
    
    /// Retun the ID of the task.
    types::subdomain_id get_subdomain_id() const;

    /// Return the value of the angular flux associated to given dof.
    double get_required_angular_flux(types::global_dof_index dof) const;

    /// Return a pointer to required_tasks.
    std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
      std::vector<types::global_dof_index>,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>>* get_required_tasks();

    /// Return a pointer to waiting_subdomains.
    std::unordered_map<types::subdomain_id,std::vector<types::global_dof_index>>*
      const get_waiting_subdomains() const;

    /// Return a pointer to the dofs of the ith element of waiting_tasks.
    std::vector<types::global_dof_index> const* get_waiting_tasks_dofs(
        unsigned int i);

    /// Return a pointer to the dofs of the given required task.
    std::vector<types::global_dof_index>* const get_required_dofs(
        std::pair<types::global_dof_index,unsigned int> const &current_task) const;

    /// Return a pointer to sweep_order.
    std::vector<unsigned int> const* get_sweep_order() const;

    /// Return a pointer to the dofs of the ith element of
    /// incomplete_required_dofs.
    std::vector<types::global_dof_index> const* get_incomplete_dofs(unsigned int i) 
      const;

  private :
    /// Direction associated to the task.
    unsigned int idir;
    /// Task Id not unique globally but unique per processor.
    unsigned int id;
    /// Number of required dofs before the task is ready.
    unsigned int n_required_dofs;
    /// Current number of required dofs that are still missing before 
    /// the task is ready. Because of the Trilinos interface in Epetra_Operator, 
    /// n_missing_dofs is made mutable so it can be changed in a const function.
    mutable unsigned int n_missing_dofs;
    /// Subdomain ID associated to the task.
    types::subdomain_id subdomain_id;
    /// Map of the required task for this task to be ready using for key the 
    /// subdomain_id and the task id of the required task and for value the 
    /// required dofs indices.
    mutable std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
      std::vector<types::global_dof_index>,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>> required_tasks;
    /// Map of the tasks waiting this task to be done using for key the 
    /// subdomain_id and the task id of the waiting task and for value the dofs 
    /// indices that are waited. 
    std::unordered_map<std::pair<types::subdomain_id,unsigned int>,
      std::vector<types::global_dof_index>,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>> waiting_tasks;
    /// Subdomains waiting angular fluxes from this task, dofs have to be
    /// sorted
    std::unordered_map<types::subdomain_id,std::vector<types::global_dof_index>>
      waiting_subdomains;
    /// Map using for key the required dof indices and for values the angular
    /// flux associated to the indices. Because of the Trilinos interface in 
    /// Epetra_Operator, required_dofs is made mutable so it can be changed 
    /// in a const function.
    mutable std::unordered_map<types::global_dof_index,double> required_dofs;
    /// Vector containing the order in which the cells of this task are swept.
    std::vector<unsigned int> sweep_order;
    /// Temporary data used to build the required_tasks map.
    std::vector<std::pair<types::subdomain_id,std::vector<types::global_dof_index>>> 
      incomplete_required_tasks;
};

inline void Task::add_to_waiting_tasks(std::pair<types::subdomain_id,unsigned int> 
    &subdomain_task_pair,types::global_dof_index dof)
{
  waiting_tasks[subdomain_task_pair].push_back(dof);
}

inline void Task::add_to_waiting_subdomains(types::subdomain_id other_subdomain_id,
    types::global_dof_index dof)
{
  waiting_subdomains[other_subdomain_id].push_back(dof);
}

inline void Task::clear_required_dofs() const
{
  required_dofs.clear();
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

inline double Task::get_required_angular_flux(types::global_dof_index dof) const
{
  return required_dofs[dof];
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

#endif
