module phase_effects
  use omp_lib
  implicit none
  real(8), parameter :: pi          = 4.d0*atan(1.d0)
  real(8), parameter :: inv_pi      = 1.d0/pi
  real(8), parameter :: half_pi     = 2.d0*atan(1.d0)
  real(8), parameter :: log_two_pi  = log(8.d0*atan(1.d0))
contains

  subroutine phase(true_anomaly, i, w, nthreads, npt, ph)
    integer, intent(in) :: npt, nthreads
    real(8), intent(in) :: i, w
    real(8), intent(in),  dimension(npt) :: true_anomaly
    real(8), intent(out), dimension(npt) :: ph
    
    !$ call omp_set_num_threads(nthreads)
    !$omp parallel workshare
    ph = sin(pi+w+true_anomaly)*sin(i)
    ph = acos(ph)
    !$omp end parallel workshare
  end subroutine phase
  
  subroutine cos_phase(true_anomaly, i, w, nthreads, npt, cph)
    integer, intent(in) :: npt, nthreads
    real(8), intent(in) :: i, w
    real(8), intent(in),  dimension(npt) :: true_anomaly
    real(8), intent(out), dimension(npt) :: cph

    !$ call omp_set_num_threads(nthreads)
    !$omp parallel workshare
    cph = sin(pi+w+true_anomaly)*sin(i)
    !$omp end parallel workshare
  end subroutine cos_phase


  subroutine cos_phase2(true_anomaly, i, w, nthreads, npt, cph)
    integer, intent(in) :: npt, nthreads
    real(8), intent(in) :: i, w
    real(8), intent(in),  dimension(npt) :: true_anomaly
    real(8), intent(out), dimension(npt) :: cph
    real(8) :: si
    integer :: j

    si = sin(i)
    !$ call omp_set_num_threads(nthreads)
    !$omp parallel do shared(true_anomaly, si, w, npt, cph) private(j)
    do j=1,npt
       cph(j) = sin(pi+w+true_anomaly(j))*si
    end do
    !$omp end parallel do
  end subroutine cos_phase2
  
  !! From Lillo-Box, J., Barrado, D., Moya, A., Montesinos, B., Montalbán, J., Bayo, A., … Henning, T. (2014). 
  !! Kepler-91b: a planet at the end of its life. Astronomy & Astrophysics, 562, A109. 
  !! doi:10.1051/0004-6361/201322001
  subroutine ellipsoidal_variation_mr(true_anomaly, phase, mass_ratio, a, i, e, u, v, nthreads, npt, ev)
    integer, intent(in) :: npt, nthreads
    real(8), intent(in) :: mass_ratio, a, i, e, u, v
    real(8), intent(in),  dimension(npt) :: true_anomaly, phase
    real(8), intent(out), dimension(npt) :: ev
    real(8) :: ae, ia3,sini2
    integer :: j

    ae    = 0.15d0*(15.d0+u)*(1.d0+v)/(3.-u)
    ia3   = 1.d0/a**3
    sini2 = sin(i)**2
    !$ call omp_set_num_threads(nthreads)
    !$omp parallel do shared(ae,ia3,sini2,e,mass_ratio,true_anomaly,phase,ev) private(j)
    do j=1,npt
       ev(j) = -ae*mass_ratio*ia3*((1.d0+e*cos(true_anomaly(j))) / (1.d0-e**2))**3 * sini2 * cos(2.d0*phase(j))
    end do
    !$omp end parallel do
  end subroutine ellipsoidal_variation_mr

  subroutine ellipsoidal_variation_a(true_anomaly, phase, amplitude, e, nthreads, npt, ev)
    integer, intent(in) :: npt, nthreads
    real(8), intent(in) :: amplitude, e
    real(8), intent(in),  dimension(npt) :: true_anomaly, phase
    real(8), intent(out), dimension(npt) :: ev
    integer :: j

    !$ call omp_set_num_threads(nthreads)
    !$omp parallel do shared(amplitude,e,true_anomaly,phase,ev) private(j)
    do j=1,npt
       ev(j) = -amplitude*((1.d0+e*cos(true_anomaly(j))) / (1.d0-e**2))**3 * cos(2.d0*phase(j))
    end do
    !$omp end parallel do
  end subroutine ellipsoidal_variation_a

  subroutine ellipsoidal_variation2(phase, amplitude, a, i, u, v, nthreads, npt, ev)
    integer, intent(in) :: npt, nthreads
    real(8), intent(in) :: amplitude, a, i, u, v
    real(8), intent(in),  dimension(npt) :: phase
    real(8), intent(out), dimension(npt) :: ev
    real(8) :: a1, f1, f2
    integer :: j

    a1 = (25.d0*u)/(24.d0*(15.d0+u)) * ((v+2.d0)/(v+1.d0))
    f1 = 3.d0*a1/a * ((5.d0*sin(i)**2-4.d0)/sin(i))
    f2 = 5.d0*a1/a * sin(i)

    !$ call omp_set_num_threads(nthreads)
    !$omp parallel do shared(a1,f1,f2,amplitude,phase,ev) private(j)
    do j=1,npt
       ev(j) = -amplitude*(cos(2.d0*phase(j)) + f1*cos(phase(j)) + f2*cos(3.d0*phase(j)))
    end do
    !$omp end parallel do
  end subroutine ellipsoidal_variation2


  subroutine lambert_phase_function(phase, nthreads, npt, pf)
    integer, intent(in) :: npt, nthreads
    real(8), intent(in),  dimension(npt) :: phase
    real(8), intent(out), dimension(npt) :: pf
    integer :: j

    !$ call omp_set_num_threads(nthreads)
    !$omp parallel do shared(phase,pf) private(j)
    do j=1,npt
       pf(j) = (sin(phase(j)) + (pi-phase(j))*cos(phase(j)))*inv_pi
    end do
    !$omp end parallel do
  end subroutine lambert_phase_function
end module phase_effects
