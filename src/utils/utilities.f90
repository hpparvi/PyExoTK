module utilities
  implicit none

  real(8), parameter :: PI      = 3.1415927d0
  real(8), parameter :: TWO_PI  = 6.2831853d0
  real(8), parameter :: HALF_PI = 1.5707963d0

contains

  subroutine downsample(x,mask,n,nx,res)
    use omp_lib
    implicit none

    integer, intent(in) :: n, nx
    real(8), intent(in) :: x(nx)
    logical, intent(in) :: mask(nx)
    real(8), intent(out), dimension(nx/n) :: res
    integer :: i,j, nres, ns

    nres = nx/n
    res = 0.0

    do i=1,nres
       ns = 0
       do j=1,n
          if (mask(i*n+j) .eqv. .true.) then
             res(i) = res(i)+x(i*n+j)
             ns = ns+1
          end if
       end do
       if (ns>0) then
          res(i) = res(i) / real(ns, 8)
       end if
    end do

  end subroutine downsample

  subroutine bin(n, x, y, nbins, xb, yb, ye)
    use omp_lib
    implicit none

    integer :: n, nbins
    real(8), dimension(n), intent(in) :: x, y
    real(8), dimension(nbins), intent(out) :: xb, yb, ye
    integer, dimension(n) :: bid, bweight
    logical, dimension(n) :: mask

    real(8) :: xw, bw, xl, xh
    integer :: i, nb

    xl = minval(x)
    xh = maxval(x)
    xw = xh - xl + 1e-8
    bw = xw / real(nbins,8)

    xb  = ([(i, i=0,nbins-1)] + 0.5) * bw + xl
    yb  = 0.
    bweight  = 0
    bid = floor((x - xl) / bw) + 1

    !$omp parallel do shared(nbins, bid, bweight, yb, ye) private(i, mask, nb)
    do i=1,nbins
       mask = bid == i
       nb = count(mask)
       bweight(i) = nb
       yb(i) = sum(y, mask=mask) / real(nb,8)
       ye(i) = sqrt(sum((pack(y,mask) - yb(i))**2) / real(nb*(nb-1),8))
    end do
    !$omp end parallel do
  end subroutine bin

end module utilities
