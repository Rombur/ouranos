/* Copyright (c) 2013, Bruno Turcksin
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Pleas refer to the file
 * license.txt for the text and further information on this license.
 */

#include "Quadrature.hh"

Quadrature::Quadrature(unsigned int sn_,unsigned int L_max_,bool galerkin_) :
  galerkin(galerkin_),
  sn(sn_),
  L_max(L_max_)
{
  // Assume that the quadrature is triangular
  n_dir = sn*(sn+2)/2;
  omega.resize(n_dir,Teuchos::SerialDenseVector<int,double> (3));
  if (galerkin==true)
    n_mom = n_dir;
  else 
    n_mom = (L_max+1)*(L_max+2)/2;
  weight.size(n_dir);
  M2D.shape(n_dir,n_mom);
}

void Quadrature::build_quadrature(const double weight_sum)
{
  // Compute sin_theta octant
  build_octant();

  // Compute omega in the other octant by deploying the octant
  deploy_octant();

  // Compute the spherical harmonics
  compute_harmonics(weight_sum);

  // Compute D
  if (galerkin==true)
  {
    int info;
    int ipiv[n_dir];
    Teuchos::LAPACK<int,double> lapack;
    Teuchos::SerialDenseMatrix<int,double> work(n_dir,n_dir);
    D2M = M2D;
    
    // Compute an LU factorization of a general M-by-N matrix using partial
    // pivoting with row interchanges
    lapack.GETRF(n_dir,n_dir,D2M.values(),D2M.stride(),ipiv,&info);
    // Compute the inverse of a matrix using the LU factorization computed by
    // GETRF
    lapack.GETRI(n_dir,D2M.values(),D2M.stride(),ipiv,work.values(),work.stride(),
        &info);
  }
  else
  {
    Teuchos::BLAS<int,double> blas;
    Teuchos::SerialDenseMatrix<int,double> weight_matrix(n_dir,n_dir);
    for (unsigned int i=0; i<n_dir; ++i)
      weight_matrix(i,i) = weight_sum*weight(i);
    D2M.shape(n_mom,n_dir);

    blas.GEMM(Teuchos::TRANS,Teuchos::NO_TRANS,n_mom,n_dir,n_dir,1.,
        M2D.values(),M2D.stride(),weight_matrix.values(),weight_matrix.stride(),
        0.,D2M.values(),D2M.stride());
  }  
}

void Quadrature::deploy_octant()
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
        weight(j+offset) = weight(j);
      omega[j+offset](2) = omega[j](2);
      switch(i)
      {
        case 1:
          omega[j+offset](0) = omega[j](0);
          omega[j+offset](1) = -omega[j](1);
          break;
        case 2 :
          omega[j+offset](0) = -omega[j](0);
          omega[j+offset](1) = -omega[j](1);
          break;
        case 3 :
          omega[j+offset](0) = -omega[j](0);
          omega[j+offset](1) = omega[j](1);
      }
    }
  }
  if (galerkin==false)
  {
    double sum_weight(0.);
    for (unsigned int i=0; i<n_dir_octant; ++i)
      sum_weight += weight(i);
    weight *= 0.25/sum_weight; 
  }
}

void Quadrature::compute_harmonics(const double weight_sum)
{
  const unsigned int L_max_x(L_max+1);
  d_vector phi(n_dir,0.);
  std::vector<std::vector<d_vector> > Ye(L_max_x,std::vector<d_vector>(L_max_x,d_vector(n_dir,0.)));
  std::vector<std::vector<d_vector> > Yo(L_max_x,std::vector<d_vector>(L_max_x,d_vector(n_dir,0.)));

  // Compute the real spherical harmonics
  for (unsigned int i=0; i<n_dir; ++i)
  {
    phi[i] = atan(omega[i](1)/omega[i](0));
    if (omega[i](0)<0.)
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
          boost::math::spherical_harmonic_r<double,double>(l,m,omega[idir](2),phi[idir]);
        Yo[l][m][idir] = w_sph*
          boost::math::spherical_harmonic_i<double,double>(l,m,omega[idir](2),phi[idir]);
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
            M2D(idir,pos) = Ye[l][m][idir];
            moment_to_order.push_back(l);
            ++pos;
          }
        }
        for (unsigned int m=1; m<=l; ++m)
        {
          // Do not use the ODD spherical harmonics when m+l is odd for l<=sn
          if ((l<=sn) && ((m+l)%2==0))
          {
            M2D(idir,pos) = Yo[l][m][idir];
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
            M2D(idir,pos) = Ye[l][m][idir];
            moment_to_order.push_back(l);
            ++pos;
          }
        }
        for (unsigned int m=1; m<=l; ++m)
        {
          // Do not use the ODD when m_l is odd
          if ((m+l)%2==0)
          {
            M2D(idir,pos) = Yo[l][m][idir];
            moment_to_order.push_back(l);
            ++pos;
          }
        }
      }
    }
  }
}
