/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "Task.hh"

Task::Task(unsigned int idir,int id,ui_vector &sweep_order,
    std::vector<std::pair<types::subdomain_id,
    std::vector<types::global_dof_index>>> &incomplete_required_tasks) :
  done(false),
  idir(idir),
  id(id),
  sweep_order(sweep_order),
  incomplete_required_tasks(incomplete_required_tasks)
{}
