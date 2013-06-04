/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Pleas refer to the file
 * license.txt for the text and further information on this license.
 */

#include <iostream>


int main(int argc,char **argv)
{  
  try
  {
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
