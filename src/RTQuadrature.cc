/* Copyright (c) 2013, Bruno Turcksin.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file
 * license.txt for the text and further information on this license.
 */

#include "RTQuadrature.hh"

RTQuadrature::RTQuadrature(unsigned int sn_,unsigned int L_max_,bool galerkin_) :
  galerkin(galerkin_),
  sn(sn_),
  L_max(L_max_)
{
  // Assume that the quadrature is triangular
  n_dir = sn*(sn+2)/2;
  omega.resize(n_dir,Vector<double>(3));
  if (galerkin==true)
    n_mom = n_dir;
  else 
    n_mom = (L_max+1)*(L_max+2)/2;
  weight.reinit(n_dir);
  M2D.reinit(n_dir,n_mom);
  D2M.reinit(n_mom,n_dir);
}

void RTQuadrature::build_quadrature(const double weight_sum)
{
  // Compute sin_theta octant
  build_octant();

  // Compute omega in the other octant by deploying the octant
  deploy_octant();

  // Compute the spherical harmonics
  compute_harmonics(weight_sum);

  // Compute D
  if (galerkin==true)
    D2M.invert(M2D);
  else
  {
    FullMatrix<double> weight_matrix(n_dir,n_dir);
    for (unsigned int i=0; i<n_dir; ++i)
      weight_matrix(i,i) = weight_sum*weight[i];

    M2D.Tmmult(D2M,weight_matrix);
  }  
}

void RTQuadrature::deploy_octant()
{
  // Assume the quadrature is only for 2D
  const unsigned int n_dir_octant(n_dir/4);
  for (unsigned int i=1; i<4; ++i)
  {
    const unsigned int offset(i*n_dir_octant);
    for (unsigned int j=0; j<n_dir_octant; ++j)
    {
      // Copy omega and weight
      if (galerkin==false)
        weight[j+offset] = weight[j];
      omega[j+offset][2] = omega[j][2];
      switch(i)
      {
        case 1:
          omega[j+offset][0] = omega[j][0];
          omega[j+offset][1] = -omega[j][1];
          break;
        case 2 :
          omega[j+offset][0] = -omega[j][0];
          omega[j+offset][1] = -omega[j][1];
          break;
        case 3 :
          omega[j+offset](0) = -omega[j][0];
          omega[j+offset](1) = omega[j][1];
      }
    }
  }
  if (galerkin==false)
  {
    double sum_weight(0.);
    for (unsigned int i=0; i<n_dir_octant; ++i)
      sum_weight += weight[i];
    for (unsigned int i=0; i<weight.size(); ++i)
      weight[i] *= 0.25/sum_weight; 
  }
}

void RTQuadrature::compute_harmonics(const double weight_sum)
{
  const unsigned int L_max_x(L_max+1);
  std::vector<double> phi(n_dir,0.);
  std::vector<std::vector<std::vector<double>>> Ye(L_max_x,
      std::vector<std::vector<double>>(L_max_x,std::vector<double>(n_dir,0.)));
  std::vector<std::vector<std::vector<double>>> Yo(L_max_x,
      std::vector<std::vector<double>>(L_max_x,std::vector<double>(n_dir,0.)));

  // Compute the real spherical harmonics
  for (unsigned int i=0; i<n_dir; ++i)
  {
    phi[i] = atan(omega[i][1]/omega[i][0]);
    if (omega[i][0]<0.)
      phi[i] += M_PI;
  }

  for (unsigned int l=0; l<L_max_x; ++l)
  {
    for (unsigned int m=0; m<l+1; ++m)
    {
      for (unsigned int idir=0; idir<n_dir; ++idir)
      {
        // If the sum of the weight is not 4pi the weight must be modified
        const double w_sph(sqrt((4.*M_PI)/weight_sum));
        // Compute the normalized associated Legendre polynomial using boost:
        // sqrt((2l+1)/4pi (l-m)!/(l+m)!) P_l^m(cos(theta)). The
        // Condon-Shortley phase is included in the associated Legendre
        // polynomial.
        Ye[l][m][idir] = w_sph*
          boost::math::spherical_harmonic_r<double,double>(l,m,omega[idir][2],phi[idir]);
        Yo[l][m][idir] = w_sph*
          boost::math::spherical_harmonic_i<double,double>(l,m,omega[idir][2],phi[idir]);
      } 
    }
  }
 
  // Build the M2D matrix
  if (galerkin==true)
  {
    for (unsigned int idir=0; idir<n_dir; ++idir)
    {
      unsigned int pos(0);
      for (unsigned int l=0; l<L_max_x; ++l)
      {
        for (int m=l; m>=0; --m)
        {
          // Do not use the EVEN spherical harmonics when m+l is odd for L<sn
          // or L=sn and m=0
          if ((l<sn) && ((m+l)%2==0))
          {
            M2D.set(idir,pos,Ye[l][m][idir]);
            moment_to_order.push_back(l);
            ++pos;
          }
        }
        for (unsigned int m=1; m<=l; ++m)
        {
          // Do not use the ODD spherical harmonics when m+l is odd for l<=sn
          if ((l<=sn) && ((m+l)%2==0))
          {
            M2D.set(idir,pos,Yo[l][m][idir]);
            moment_to_order.push_back(l);
            ++pos;
          }
        }
      }
    }
  }
  else
  {
    for (unsigned int idir=0; idir<n_dir; ++idir)
    {
      unsigned int pos(0);
      for (unsigned int l=0; l<L_max_x; ++l)
      {
        for (int m=l; m>=0; --m)
        {
          // Do not use the EVEN spherical harmonics when m+l is odd
          if ((m+l)%2==0)
          {
            M2D.set(idir,pos,Ye[l][m][idir]);
            moment_to_order.push_back(l);
            ++pos;
          }
        }
        for (unsigned int m=1; m<=l; ++m)
        {
          // Do not use the ODD when m_l is odd
          if ((m+l)%2==0)
          {
            M2D.set(idir,pos,Yo[l][m][idir]);
            moment_to_order.push_back(l);
            ++pos;
          }
        }
      }
    }
  }
}
