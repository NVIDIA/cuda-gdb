! Copyright 2006, 2010 Free Software Foundation, Inc.
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <http://www.gnu.org/licenses/>.
!
! Ihis file is the Fortran source file for derived-type.exp.  It was written
! by David Lecomber (david@allinea.com)

program main

  type bar
    integer :: c
    real :: d
    integer :: x(3,5)
    integer, allocatable :: y(:,:)
  end type bar

  type(bar) :: p
  type(bar), allocatable :: q (:)
  integer :: i, j

  i = 1
  j = 1
  ! starting break
  allocate (q(1))
  ! alloc q base
  allocate (q(1)%y(3,5))
  ! alloc q y content
  allocate (p%y(3,5))
  ! alloc p y content

  do i = 1,3
     do j = 1,5
        q(1)%x(i,j) = i * 5 + j
        q(1)%y(i,j) = i * 5 + j
        p%x(i,j) = i * 5 + j
        p%y(i,j) = i * 5 + j
     end do
  end do


  print *, p%y    ! A great place to stop and grab a beer
  print *, q(1)%y 

end program main
