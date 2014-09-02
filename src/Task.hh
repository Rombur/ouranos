/* Copyright (c) 2013, 2014 Bruno Turcksin.
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
#include <unordered_set>
#include <tuple>
#include <utility>
#include <vector>
#include "boost/functional/hash/hash.hpp"
#include "deal.II/base/exceptions.h"
#include "deal.II/base/types.h"

using namespace dealii;


class Task
{
  public :
    typedef std::pair<types::subdomain_id,std::vector<types::global_dof_index>> domain_pair;
    typedef std::tuple<types::subdomain_id,unsigned int,std::vector<types::global_dof_index>> task_tuple; 

    Task(unsigned int idir,unsigned int id,types::subdomain_id subdomain_id,
        std::vector<unsigned int> &sweep_order,
        std::vector<std::pair<types::subdomain_id,
        std::vector<types::global_dof_index>>> &incomplete_required_tasks);

    /// Add a dof index to the given required task.
    void add_to_required_tasks(types::subdomain_id s_id,unsigned int t_id,types::global_dof_index* recv,
        unsigned int pos,unsigned int n);

    /// Add a dof index to the given waiting task.
    void add_to_waiting_tasks(task_tuple const &tmp_task);

    /// Add a dof index to the given waiting subdomain.
    void add_to_waiting_subdomains(domain_pair const &waiting_subdomains_dofs);

    /// Clear the required_dofs map.
    void clear_required_dofs() const;

    /// Suppress duplicated wating tasks that have the same subdomain_id and
    /// accumate the dofs in the unique tasks.
    void compress_waiting_tasks();

    /// Sort the dofs associated to waiting processors (subdomains) and
    /// suppress duplicates.
    void compress_waiting_subdomains();

    /// Create local_waiting_tasks from waiting_tasks and required_tasks_map
    /// from required_tasks_map then call clear_temporary_data.
    void finalize_maps();

    /// Output the data for debug purpose.
    void print(std::ostream &output_stream);

    /// Set the given angular flux value to the given dof index when the value
    /// has been computed by another task.
    void set_required_dof(types::global_dof_index dof,double value) const;

    /// Set the given angular flux value to the given dof index when the value
    /// has been computed by the current task.
    void set_local_required_dof(types::global_dof_index dof,double value) const;

    /// Return true if the task is ready for sweep.
    bool is_ready() const;

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

    /// Return the subdomain ID of the ith element of
    /// incomplete_required_tasks.
    types::global_dof_index get_incomplete_subdomain_id(unsigned int i) const;

    /// Return the subdomain ID of the ith element in waiting_tasks.
    types::subdomain_id get_waiting_tasks_subdomain_id(unsigned int i) const;
    
    /// Retun the ID of the task.
    types::subdomain_id get_subdomain_id() const;

    /// Return the value of the angular flux associated to given dof.
    double get_required_angular_flux(types::global_dof_index dof) const;

    /// Return a const_iterator to the beginning of the required_tasks
    /// vector.
    std::vector<task_tuple>::const_iterator get_required_tasks_cbegin() const;

    /// Return a const_iterator to the end of the required_tasks vector.
    std::vector<task_tuple>::const_iterator get_required_tasks_cend() const;

    /// Return a const_iterator to the beginning of the waiting_subdomains
    /// vector.
    std::vector<domain_pair>::const_iterator get_waiting_subdomains_cbegin() const;

    /// Return a const_iterator to the end of the waiting_subdomains vector.
    std::vector<domain_pair>::const_iterator get_waiting_subdomains_cend() const;

    /// Return a const_iterator to the beginning of the local_waiting_tasks
    /// vector.
    std::vector<std::pair<unsigned int,std::vector<types::global_dof_index>>>
      ::const_iterator get_local_waiting_tasks_cbegin() const;

    /// Return a const_iterator to the end of the local_waiting_tasks vector.
    std::vector<std::pair<unsigned int,std::vector<types::global_dof_index>>>
      ::const_iterator get_local_waiting_tasks_cend() const;

    /// Return a pointer to the dofs of the ith element of waiting_tasks.
    std::vector<types::global_dof_index> const* get_waiting_tasks_dofs(
        unsigned int i);

    /// Return a pointer to the dofs of the given required task.
    std::vector<types::global_dof_index>* const get_required_dofs(
        types::subdomain_id required_subdomain_id,unsigned int required_task_id) const;

    /// Return a pointer to sweep_order.
    std::vector<unsigned int> const* get_sweep_order() const;

    /// Return a pointer to the dofs of the ith element of
    /// incomplete_required_dofs.
    std::vector<types::global_dof_index> const* get_incomplete_dofs(unsigned int i) 
      const;

  private :
    /// Clear incomplete_required_tasks and waiting_tasks.
    void clear_temporary_data();

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
    /// Vector of the tasks required for the current task to be executed. Each
    /// element of the vector is a tuple of subdomain_id, task id, and dof 
    /// indices.
    std::vector<task_tuple> required_tasks;
    /// Map used to find the position of a given task in the required_tasks
    /// vector (the key is the subdomain_id and the task id of the
    /// required task and the value is the position in the vector).
    std::unordered_map<std::pair<types::subdomain_id,unsigned int>,unsigned int,
      boost::hash<std::pair<types::subdomain_id,unsigned int>>> required_tasks_map;
    /// Vector of the tasks waiting for the current task to be done. Each element of the
    /// vector is a tuple of subdomain_id, task id, and dof indices. This
    /// vector is a temporary vector used to during the setup.
    std::vector<task_tuple> waiting_tasks;
    /// Vector of the waiting tasks that are the same processor as the current task.
    std::vector<std::pair<unsigned int,std::vector<types::global_dof_index>>> local_waiting_tasks;
    /// Subdomains waiting angular fluxes from this task, dofs have to be
    /// sorted
    std::vector<domain_pair> waiting_subdomains;
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

inline void Task::add_to_waiting_tasks(task_tuple const &tmp_task)
{
  waiting_tasks.push_back(tmp_task);
}

inline void Task::add_to_waiting_subdomains(domain_pair const &waiting_subdomains_dofs)
{
  waiting_subdomains.push_back(waiting_subdomains_dofs);
}

inline void Task::clear_required_dofs() const
{
  required_dofs.clear();
}

inline void Task::set_required_dof(types::global_dof_index dof,double value) const
{
  required_dofs[dof] = value;
  --n_missing_dofs;
}

inline void Task::set_local_required_dof(types::global_dof_index dof,double value) const
{
  required_dofs[dof] = value;
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

inline unsigned int Task::get_waiting_tasks_id(unsigned int i) const
{
  AssertIndexRange(i,waiting_tasks.size());
  return std::get<1>(waiting_tasks[i]);
}

inline unsigned int Task::get_waiting_tasks_n_dofs(unsigned int i) const
{
  AssertIndexRange(i,waiting_tasks.size());
  return std::get<2>(waiting_tasks[i]).size();
}

inline types::subdomain_id Task::get_incomplete_subdomain_id(unsigned int i) const
{
  AssertIndexRange(i,incomplete_required_tasks.size());
  return incomplete_required_tasks[i].first;
}

inline types::subdomain_id Task::get_waiting_tasks_subdomain_id(unsigned int i) const
{
  AssertIndexRange(i,waiting_tasks.size());
  return std::get<0>(waiting_tasks[i]);
}

inline unsigned int Task::get_incomplete_n_dofs(unsigned int i) const
{
  AssertIndexRange(i,incomplete_required_tasks.size());
  return incomplete_required_tasks[i].second.size();
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

inline std::vector<Task::domain_pair>::const_iterator Task::get_waiting_subdomains_cbegin() const
{
  return waiting_subdomains.cbegin();
}

inline std::vector<Task::domain_pair>::const_iterator Task::get_waiting_subdomains_cend() const
{
  return waiting_subdomains.cend();
}

inline std::vector<std::pair<unsigned int,std::vector<types::global_dof_index>>>::const_iterator 
Task::get_local_waiting_tasks_cbegin() const
{
  return local_waiting_tasks.cbegin();
}

inline std::vector<std::pair<unsigned int,std::vector<types::global_dof_index>>>::const_iterator 
Task::get_local_waiting_tasks_cend() const
{
  return local_waiting_tasks.cend();
}

inline std::vector<types::global_dof_index> const* Task::get_waiting_tasks_dofs(
    unsigned int i)
{
  AssertIndexRange(i,waiting_tasks.size());

  return &(std::get<2>(waiting_tasks[i]));
}

inline std::vector<Task::task_tuple>::const_iterator Task::get_required_tasks_cbegin() const
{
  return required_tasks.cbegin();
}
    
inline std::vector<Task::task_tuple>::const_iterator Task::get_required_tasks_cend() const
{
  return required_tasks.cend();
}

inline std::vector<unsigned int> const* Task::get_sweep_order() const
{
  return &sweep_order;
}

inline std::vector<types::global_dof_index> const* Task::get_incomplete_dofs(
    unsigned int i) const
{
  AssertIndexRange(i,incomplete_required_tasks.size());
  return &(incomplete_required_tasks[i].second);
}

inline std::vector<types::global_dof_index>* const 
Task::get_required_dofs(types::subdomain_id required_subdomain_id,unsigned int required_task_id) const
{   
  // Dont use iterator because need to do the search twice
  // In map need find because of const
  const unsigned int pos(required_tasks_map.find(std::pair<types::subdomain_id,unsigned int> 
      (required_subdomain_id,required_task_id))->second);
  return const_cast<std::vector<types::global_dof_index>* const> (&(std::get<2>(required_tasks[pos])));
}

#endif
