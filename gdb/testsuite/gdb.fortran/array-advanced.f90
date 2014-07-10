! Copyright 2005, 2010 Free Software Foundation, Inc.

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

! This file is the F90 source file for array-advanced.exp.  It was written
! by David Lecomber (david@allinea.com).

program main
  integer :: a(10), b(11, 13), c(13,15,17), d(4:6,8:12)
  integer :: i, j, k
  integer, allocatable :: oned(:), twod(:,:), threed(:,:,:)
  
  call fillone(a, 10)
  call filltwo(b, 11, 13)
  call fillthree(c, 13, 15, 17)

  call testpoint1()

  allocate(oned(10))

  allocate(twod(11, 13))

  allocate(threed(13,15,17))

  call testpoint2()

  call fillone(oned, 10)
  call filltwo(twod, 11, 13)
  call fillthree(threed, 13, 15, 17)

  call testpoint3()

  deallocate(oned)
  deallocate(twod)
  deallocate(threed)

  call testpoint4()

  write(*,*) 'This is a test.'

  write(*,*) a
  write(*,*) b
  write(*,*) c
!  write(*,*) d
  stop
end program main

subroutine fillone(a, n)
  integer :: a(n)
  integer :: i
  do i = 1, n
     a(i) = i
  end do
end subroutine fillone

subroutine filltwo(b, n, m)
  integer :: b(n, m)
  integer :: i, j
  do i = 1, n
     do j = 1, m
        b(i, j) = i * m + j
     end do
  end do
end subroutine filltwo

subroutine fillthree(c, n, m, o)
  integer :: n, m, o
  integer :: c(n, m, o)
  integer :: i, j, k

  do i = 1, n
     do j = 1, m
        do k = 1, o
           c(i, j, k) = i * m * o + j * o + k
        end do
     end do
  end do
end subroutine fillthree

  
subroutine testpoint1()
  write (*, *) 'At testpoint1'
end subroutine testpoint1

subroutine testpoint2()
  write (*, *) 'At testpoint2'
end subroutine testpoint2

subroutine testpoint3()
  write (*, *) 'At testpoint3'
end subroutine testpoint3

subroutine testpoint4()
  write (*, *) 'At testpoint4'
end subroutine testpoint4
