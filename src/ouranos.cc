/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Pleas refer to the file
 * license.txt for the text and further information on this license.
 */

#include <iostream>
#include "deal.II/base/utilities.h"
#include "deal.II/base/mpi.h"
#include "deal.II/base/logstream.h"

using namespace dealii;

int main(int argc,char **argv)
{  
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv);

  try
  {
    // Suppress output on the screen
    deallog.depth_console(0);
    
  }
  catch(std::exception &exc)
  {
    std::cerr<<std::endl<<std::endl
             <<"-----------------------------------"
             <<std::endl;
    std::cerr<<"Exception on processing: "<<std::endl
             <<exc.what()<<std::endl
             <<std::endl
             <<"Aborting!"<<std::endl
             <<"-----------------------------------"
             <<std::endl;

    return 1;
  }
  catch(...)
  {
    std::cerr<<std::endl<<std::endl
             <<"-----------------------------------"
             <<std::endl;
    std::cerr<<"Unknown exception!" <<std::endl
             <<"Aborting!"<<std::endl
             <<"-----------------------------------"
             <<std::endl;

    return 1;
  }

  return 0;
}
