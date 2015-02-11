/* Copyright (c) 2015, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _SCHEDULERFACTORY_HH_
#define _SCHEDULERFACTORY_HH_

#include "CAPPFBScheduler.hh"
#include "Parameters.hh"
#include "RandomScheduler.hh"
#include "Scheduler.hh"

// Forward declaration of ConditionalOStream, Epetra_MpiComm, and RTQuadrature 
// since they are not use by SchedulerFactory directly.
namespace dealii
{
  class ConditionalOStream;
}
class Epetra_MpiComm;
class RTQuadrature;

using namespace dealii;

/**
 * The only purpose of this class is to create a scheduler using a factory
 * method.
 */
class SchedulerFactory
{
  public :
    /// Factory method that returns a Scheduler pointer from a CAPPFBScheduler
    /// or a RandomScheduler.
    template <int dim,int tensor_dim>
    static Scheduler<dim,tensor_dim>* create_scheduler(SCHEDULER_TYPE scheduler_type,
        RTQuadrature const* quad,Epetra_MpiComm const* comm,
        ConditionalOStream const &pcout,unsigned int max_iter)
    {
      switch (scheduler_type)
      {
        case CAP_PFB :
          {
            return new CAPPFBScheduler<dim,tensor_dim>(quad,comm,pcout,max_iter);

            break;
          }
        default :
          {
            return new RandomScheduler<dim,tensor_dim>(quad,comm,pcout);
          }
      }
    }
};

#endif

