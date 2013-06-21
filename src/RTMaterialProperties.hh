/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#ifndef _RTMATERIALPROPERTIES_HH_
#define _RTMATERIALPROPERTIES_HH_

#include <fstream>
#include <string>
#include <vector>
#include "deal.II/base/exceptions.h"
#include "deal.II/base/parameter_handler.h"

using namespace dealii;

typedef std::vector<double> d_vector;

/**
 * This class reads the radiative transfer material properties files.
 */

class RTMaterialProperties
{
  public :
    RTMaterialProperties(std::string &material_properties_filename,
        unsigned int n_materials,unsigned int n_groups);

    /// Retun the number of materials.
    unsigned int get_n_materials() const;

    /// Return the number of groups.
    unsigned int get_n_groups() const;

    /// Return L_max.
    unsigned int get_L_max() const;

    /// Return the total cross section for a given material and a given group.
    double get_sigma_t(unsigned int material_id,unsigned int g) const;

    /// Return the scattering cross sections for a given material, two given
    /// groups, and a given moment.
    double get_sigma_s(unsigned int material_id,unsigned int g,unsigned int gp,
        unsigned int mom) const;

  private :
    /// Declare the parameters and assign default values.
    void declare_parameters(ParameterHandler &prm);

    /// Parse the parameters.
    void parse_parameters(ParameterHandler &prm);
    
    /// Return a list of n_elements doubles from a given string.
    d_vector get_list_double(std::string &input,unsigned int n_elements);

    /// Number of materials.
    unsigned int n_materials;
    /// Number of groups.
    unsigned int n_groups;
    /// Number of moments.
    unsigned int L_max;
    /// Total cross sections.
    d_vector sigma_t;
    /// Scattering cross sections.
    d_vector sigma_s;
};

inline unsigned int RTMaterialProperties::get_n_materials() const
{
 return n_materials;
}

inline unsigned int RTMaterialProperties::get_n_groups() const
{
  return n_groups;
}

inline unsigned int RTMaterialProperties::get_L_max() const
{
  return L_max;
}

inline double RTMaterialProperties::get_sigma_t(unsigned int material_id,
    unsigned int g) const
{
  AssertIndexRange(material_id,n_materials);
  AssertIndexRange(g,n_groups);
  return sigma_t[material_id*n_groups+g];
}

inline double RTMaterialProperties::get_sigma_s(unsigned int material_id,
    unsigned int g,unsigned int gp,unsigned int mom) const
{
  AssertIndexRange(material_id,n_materials);
  AssertIndexRange(g,n_groups);
  AssertIndexRange(gp,n_groups);
  AssertIndexRange(mom,L_max+1);
  return sigma_s[mom+gp*(L_max+1)+g*n_groups*(L_max+1)+material_id*n_groups*
    n_groups*(L_max+1)];
}

#endif

