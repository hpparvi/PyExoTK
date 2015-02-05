module lh
  use omp_lib
  implicit none
  real(8), parameter :: pi          = 4.d0*atan(1.d0)
  real(8), parameter :: half_pi     = 2.d0*atan(1.d0)
  real(8), parameter :: log_two_pi  = log(8.d0*atan(1.d0))

  contains

    real(8) function ll_normal_es(o,m,e,npt,nthreads)
      integer, intent(in) :: npt, nthreads
      real(8), intent(in), dimension(npt) :: o, m
      real(8), intent(in) :: e
      real(8) :: chi2 
      integer :: i

      !$ call omp_set_num_threads(nthreads)
      !$omp parallel do reduction(+:chi2) shared(o,m,npt) private(i)
      do i=1,npt
         chi2 = chi2 + (o(i)-m(i))**2
      end do
      !$omp end parallel do
      chi2 = chi2/e**2

      ll_normal_es = -npt*log(e) -0.5d0*npt*log_two_pi -0.5d0*chi2
    end function ll_normal_es

    real(8) function ll_normal_ev(o,m,e,npt,nthreads)
      integer, intent(in) :: npt, nthreads
      real(8), intent(in), dimension(npt) :: o,m,e
      real(8) :: chi2, loge 
      integer :: i

      !$ call omp_set_num_threads(nthreads)
      !$omp parallel do reduction(+:chi2,loge) shared(o,m,npt)
      do i=1,npt
         chi2 = chi2 + ((o(i)-m(i))/e(i))**2
         loge = loge + log(e(i))
      end do
      !$omp end parallel do

      ll_normal_ev = -loge -0.5d0*npt*log_two_pi -0.5d0*chi2
  end function ll_normal_ev

end module lh
