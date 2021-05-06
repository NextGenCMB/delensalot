! See python code for doc
module bicubic
    implicit none
    contains
        double precision function cubicfilter(x, c0, c1, c2, c3)
            ! filter 4 values using cubic splines
            double precision, intent(in) :: x, c0, c1, c2, c3
            double precision w0, w1, w2, w3
            w0 = x*(x*(-x + 3d0) - 3d0) + 1d0
            w1 = x*x*(3d0*x - 6d0) + 4d0;
            w2 =  x*(x*(-3d0*x + 3d0) + 3d0) + 1d0
            w3 =  x*x*x
            cubicfilter = (c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3) * 0.16666666666666666d0
        end

        double precision function eval(ftl_map, fx, fy)
            ! nx : second dimension of array
            ! ny : first dimension of array
            implicit none
            double precision, intent(in) :: ftl_map(0:,0:), fx, fy
            double precision gx, gy
            integer px, py, nx, ny
            !tex2d2 = ftl_map(modulo(i, ny),modulo(j,  nx))
            nx = size(ftl_map, 2)
            ny = size(ftl_map, 1)
            px = floor(fx)
            py = floor(fy)
            gx = fx - px
            gy = fy - py
            if ((px > 0) .and. (px + 2 < ny) .and. (py > 0) .and. (py + 2 < nx)) then
                eval = cubicfilter(gy, &
                    cubicfilter(gx, ftl_map(px-1,py-1), ftl_map(px,  py-1), &
                                    ftl_map(px+1,py-1), ftl_map(px+2,py-1)), &
                    cubicfilter(gx, ftl_map(px-1,py),   ftl_map(px,  py),   &
                                    ftl_map(px+1,py),   ftl_map(px+2,py)),&
                    cubicfilter(gx, ftl_map(px-1,py+1), ftl_map(px,  py+1), &
                                    ftl_map(px+1,py+1), ftl_map(px+2,py+1)), &
                    cubicfilter(gx, ftl_map(px-1,py+2), ftl_map(px,  py+2), &
                                    ftl_map(px+1,py+2), ftl_map(px+2,py+2)))
            else
                eval = cubicfilter(gy, &
                        cubicfilter(gx, ftl_map(modulo(px-1, ny),modulo(py-1,  nx)), ftl_map(modulo(px, ny)  ,modulo(py-1,  nx)), &
                                        ftl_map(modulo(px+1, ny),modulo(py-1,  nx)), ftl_map(modulo(px+2, ny),modulo(py-1,  nx))), &
                        cubicfilter(gx, ftl_map(modulo(px-1, ny),modulo(py,  nx)),   ftl_map(modulo(px, ny)  ,modulo(py,  nx)),   &
                                        ftl_map(modulo(px+1, ny),modulo(py,  nx)),   ftl_map(modulo(px+2, ny),modulo(py,  nx))),&
                        cubicfilter(gx, ftl_map(modulo(px-1, ny),modulo(py+1,  nx)), ftl_map(modulo(px, ny)  ,modulo(py+1,  nx)), &
                                        ftl_map(modulo(px+1, ny),modulo(py+1,  nx)), ftl_map(modulo(px+2, ny),modulo(py+1,  nx))), &
                        cubicfilter(gx, ftl_map(modulo(px-1, ny),modulo(py+2,  nx)), ftl_map(modulo(px, ny)  ,modulo(py+2,  nx)), &
                                        ftl_map(modulo(px+1, ny),modulo(py+2,  nx)), ftl_map(modulo(px+2, ny),modulo(py+2,  nx)) ))

            end if
        end
end module bicubic

module remapping
    implicit none
    double precision, parameter :: DPI=3.141592653589793238d0, TOLAMIN=1e-10
    integer, parameter :: ITRMAX=1000

    interface d2ang
        module procedure d2ang_scal
    end interface d2ang
    interface ang2d
        module procedure ang2d_scal
    end interface ang2d

    contains

    subroutine get_tht(ctht, stht, tht)
        implicit None
        double precision, intent(in) :: ctht, stht  ! cost and sint
        double precision, intent(out) :: tht
        integer version

        version = nint(ctht)
            if (version == 0) then
                tht = dacos(ctht)
            else if (version == 1) then
                tht = dasin(stht)
            else if (version == -1) then
                tht = DPI - dasin(stht)
            else
                write(*, *) 'invalid version (-1, 0, 1)', version
                error stop
            end if
    end subroutine get_tht

    subroutine d2ang_scal(Red, Imd, tht, phi, thtp, phip)
        ! no phi modulo here. In principle double prec. accurate and avoiding any division by zero
        ! version -1, 0, 1 for pints close to the north pole, equator or south pol
        implicit none
        double precision, intent(in) :: Red, Imd, tht, phi
        double precision, intent(out) :: thtp, phip
        double precision d, cost, costp, sind_d, sint, sintp, e_t, e_tp, e_d
        integer version

        d = Red * Red + Imd * Imd
        if (d > 0.0001d0) then
            write(*, *) 'too large deflection here, this should not happen', dsqrt(d), tht, phi, version
            error stop
        end if
        sind_d = 1.d0 - d / 6.d0 * (1.d0 - d / 20d0 * (1.d0 - d / 42d0))
        d = dsqrt(d)
        !version = nint(cos(tht)) ! 'nearest int' 1 0 -1 for closest to north pole, equator or south pole respectively
        version = nint(1 - 2 * tht / DPI)
        if (version == 0) then
            cost = dcos(tht)
            costp = cost * dcos(d) - Red * sind_d * dsqrt(1.d0 - cost * cost)
            thtp = dacos(costp)
            phip = phi + dasin(Imd / dsqrt(1. - costp * costp) * sind_d) ! ok except for absurdly large d
            return
        end if
        e_d = 2 * dsin(d * 0.5) ** 2
        sint = dsin(tht)

        if (version == 1) then
            e_t = 2 * dsin(tht * 0.5) ** 2
            e_tp = e_t + e_d - e_t * e_d +  Red * sind_d * sint
        else if (version == -1) then
            e_t = 2 * dcos(tht * 0.5) ** 2
            e_tp = e_t + e_d - e_t * e_d -  Red * sind_d * sint
        else
            write(*, *) 'invalid version parameter (must be in (-1, 0, 1))', version
            error stop
        end if
        sintp = dsqrt(e_tp * (2 - e_tp))
        if (version ==  1) then
            thtp = dasin(sintp)
            phip = phi + datan2( Imd * sind_d, (1.d0 - e_d) * sint + Red * sind_d * (1.d0 - e_t))
        else
            thtp = DPI - dasin(sintp)
            phip = phi + datan2( Imd * sind_d, (1.d0 - e_d) * sint + Red * sind_d * (e_t - 1.d0))
        end if
        return
    end subroutine d2ang_scal

    subroutine ang2d_scal(thtp, tht, dphi, red, imd)
        implicit none
        double precision, intent(in) :: thtp, tht, dphi
        double precision, intent(out) :: red, imd
        double precision sintp, sind, norm
        sintp = dsin(thtp)
        red = dsin(thtp - tht) - 2 * dsin(dphi * 0.5d0) ** 2 * dcos(tht) * sintp  ! Red sind / d
        imd = dsin(dphi) * sintp  ! Imd sind / d
        sind = dsqrt(red * red + imd * imd)
        if (sind > 0) then
            norm = dasin(sind) / sind
            red = red * norm
            imd = imd * norm
        end if
    end subroutine ang2d_scal

    subroutine solve_pix(nt_f, np_f, ref, imf, tht, phi, tht0, phi0, t2grid, p2grid, thti, phii, redi, imdi)
        ! scalar version of solve_ring
        use bicubic, only : eval
        implicit none
        double precision, intent(in) :: tht, tht0, phi0, t2grid, p2grid
        double precision, intent(in) :: phi, ref(nt_f, np_f), imf(nt_f, np_f)
        double precision, intent(out) :: thti, phii, redi, imdi
        integer, intent(in) :: nt_f, np_f
        double precision x, y, xi, yi, xn, yn, maxres
        double precision thtn, phin
        double precision red, imd, re_res, im_res, cosp, sinp, tol
        double precision ft, fp ! tht and phi deflection in grid units
        double precision :: PI2 = DPI * 2
        integer itr

        ft = (tht - tht0) * t2grid
        fp = modulo(phi - phi0, PI2) * p2grid
        redi = -eval(ref, ft, fp) ! Starting point is -red -i imd
        imdi = -eval(imf, ft, fp)
        re_res = 0d0
        im_res = 0d0
        maxres = 10.
        itr = 0
        tol = max(TOLAMIN / 180 / 60 * DPI, 1d-15)
        do while ( (maxres >= tol) .and. (itr <= ITRMAX))
            itr = itr + 1
            call d2ang(redi, imdi, tht, phi, thti, phii)
            ft = (thti - tht0) * t2grid
            fp =  modulo(phii - phi0, PI2) * p2grid
            red = eval(ref, ft, fp)
            imd = eval(imf, ft, fp)
            call d2ang(red, imd, thti, phii, thtn, phin)
            call ang2d(thtn, tht, phin - phi, re_res, im_res) ! residual deflection field
            maxres = dsqrt(re_res * re_res + im_res * im_res)
            redi = redi - re_res
            imdi = imdi - im_res
        end do
        if (itr > itrmax) then
            write(*, *) 'redi, imdi -solver failed to achieve convergence', maxres, itr, itrmax
            if (maxres > 1d-5) then
                error stop
            end if
        end if
    end subroutine solve_pix

    subroutine solve_pixs(npix, nt_f, np_f, ref, imf, thts, phis, tht0, phi0, t2grid, p2grid, redi, imdi)
        implicit none
        integer, intent(in) :: npix, nt_f, np_f
        double precision, intent(in) :: ref(nt_f, np_f), imf(nt_f, np_f), thts(npix), phis(npix)
        double precision, intent(in) :: tht0, phi0, t2grid, p2grid
        double precision, intent(out) :: redi(npix), imdi(npix)
        double precision thti, phii, redip, imdip
        integer ip
        !$OMP PARALLEL DO DEFAULT(NONE)&
        !$OMP SHARED(thts, phis, redi, imdi, ref, imf, npix, tht0, phi0, t2grid, p2grid, np_f, nt_f)&
        !$OMP PRIVATE(ip, thti, phii, redip, imdip)
        do ip = 1, npix
            call solve_pix(nt_f, np_f, ref, imf, thts(ip), phis(ip), tht0, phi0, t2grid, p2grid, thti, phii, redip, imdip)
            redi(ip) = redip
            imdi(ip) = imdip
        end do
        !$OMP END PARALLEL DO
    end subroutine solve_pixs

end module remapping