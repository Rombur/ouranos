/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _TASK_HH_
#define _TASK_HH_

class Task
{
  public :
    Task(unsigned int idir,int id,ui_vector sweep_order);

  private :
    bool done;
    unsigned int idir;
    unsigned int weight;
    int id;
    ui_vector sweep_order;
    // Pair (task_id (positive short int), subdomain_id), dof of the flux passed
    // Required tasks before we can start this task
    std::map<std::pair<int,subdomain_id>,std::vector<types::global_dof_index>> 
      required_tasks;
    // Tasks waiting for this task to be done
    std::map<std::vector<types::global_dof_index>,std::pair<int,subdomain_id>> 
      waiting_tasks;
};

#endif
