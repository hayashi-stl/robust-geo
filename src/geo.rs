//! Contains the geometric tests

use crate::Expansion;
use crate::{square, two_product, two_sum};

use nalgebra::{Vector2, Vector3};
type Vec2 = Vector2<f64>;
type Vec3 = Vector3<f64>;

const EPSILON: f64 = f64::EPSILON / 2.0;

const ORIENT_2D_BOUND_A: f64 = (3.0 + 16.0 * EPSILON) * EPSILON;
const ORIENT_2D_BOUND_B: f64 = (2.0 + 12.0 * EPSILON) * EPSILON;
const ORIENT_2D_BOUND_C1: f64 = (3.0 + 8.0 * EPSILON) * EPSILON;
const ORIENT_2D_BOUND_C2: f64 = (9.0 + 64.0 * EPSILON) * EPSILON * EPSILON;

const ORIENT_3D_BOUND_A: f64 = (7.0 + 56.0 * EPSILON) * EPSILON;
const ORIENT_3D_BOUND_B: f64 = (3.0 + 28.0 * EPSILON) * EPSILON;
const ORIENT_3D_BOUND_C1: f64 = (3.0 + 8.0 * EPSILON) * EPSILON;
const ORIENT_3D_BOUND_C2: f64 = (26.0 + 288.0 * EPSILON) * EPSILON * EPSILON;

const IN_CIRCLE_BOUND_A: f64 = (10.0 + 96.0 * EPSILON) * EPSILON;
const IN_CIRCLE_BOUND_B: f64 = (4.0 + 48.0 * EPSILON) * EPSILON;
const IN_CIRCLE_BOUND_C1: f64 = (3.0 + 8.0 * EPSILON) * EPSILON;
const IN_CIRCLE_BOUND_C2: f64 = (44.0 + 576.0 * EPSILON) * EPSILON * EPSILON;

const IN_SPHERE_BOUND_A: f64 = (16.0 + 224.0 * EPSILON) * EPSILON;
const IN_SPHERE_BOUND_B: f64 = (5.0 + 72.0 * EPSILON) * EPSILON;
const IN_SPHERE_BOUND_C1: f64 = (3.0 + 8.0 * EPSILON) * EPSILON;
const IN_SPHERE_BOUND_C2: f64 = (71.0 + 1408.0 * EPSILON) * EPSILON * EPSILON;

/// Calculates the orientation of points `a`, `b`, `c` in a plane.
/// Returns a positive number if they define a left turn,
/// a negative number if they define a right turn,
/// and 0 if they are collinear.
pub fn orient_2d(a: Vec2, b: Vec2, c: Vec2) -> f64 {
    let diag1 = (a.x - c.x) * (b.y - c.y);
    let diag2 = (a.y - c.y) * (b.x - c.x);
    let det = diag1 - diag2;
    let det_sum = diag1.abs() + diag2.abs();

    if det.abs() >= det_sum * ORIENT_2D_BOUND_A {
        det
    } else {
        orient_2d_adapt(a, b, c, det_sum)
    }
}

fn orient_2d_adapt(a: Vec2, b: Vec2, c: Vec2, det_sum: f64) -> f64 {
    let diag1 = two_product(a.x - c.x, b.y - c.y);
    let diag2 = two_product(a.y - c.y, b.x - c.x);
    let det_hi = diag1 - diag2;
    let det_approx = det_hi.approximate();

    if det_approx.abs() >= det_sum * ORIENT_2D_BOUND_B {
        return det_approx;
    }

    // Calculate correction factor, but don't go for the full exact value yet
    let ax = two_sum(a.x, -c.x);
    let ay = two_sum(a.y, -c.y);
    let bx = two_sum(b.x, -c.x);
    let by = two_sum(b.y, -c.y);

    let det_mid1 = ax[0] * by[1] - ay[0] * bx[1];
    let det_mid2 = ax[1] * by[0] - ay[1] * bx[0];
    let det = det_mid1 + det_mid2 + det_approx;

    if det.abs() >= ORIENT_2D_BOUND_C1 * det_approx.abs() + ORIENT_2D_BOUND_C2 * det_sum {
        return det;
    }

    // Determinant is likely 0; go for exact value
    let det_hi = det_hi.dynamic();
    let det_m1 = (two_product(ax[0], by[1]) - two_product(ay[0], bx[1])).dynamic();
    let det_m2 = (two_product(ax[1], by[0]) - two_product(ay[1], bx[0])).dynamic();
    let det_lo = (two_product(ax[0], by[0]) - two_product(ay[0], bx[0])).dynamic();
    let det = (det_hi + det_m1) + (det_m2 + det_lo);
    det.highest_magnitude()
}

macro_rules! sep_let {
    (($($d:tt $var:ident),*) => let $($name:ident : [$($vals:tt)*]),* $(,)? = $($expr:tt)*) => {
        macro_rules! __mac {
            ($($d $var:ident),*) => { $($expr)* };
            ($($d $var:expr),*) => { $($expr)* };
        }

        $(let $name = __mac!($($vals)*);)*
    };
}

// xyz cyclic permutation shortcut.
// Simplifies cofactor repetition.
macro_rules! sep_xyz {
    (($dx:tt $x:ident, $dy:tt $y:ident, $dz:tt $z:ident $(, $d:tt $var:ident)*)
        => let $n1:ident : [$($v1:tt)*], $n2:ident : [$($v2:tt)*], $n3:ident : [$($v3:tt)*] = $($expr:tt)*) =>
    {
        sep_let!(($dx $x, $dy $y, $dz $z $(, $d $var)*) =>
            let $n1: [$x, $y, $z, $($v1)*], $n2: [$y, $z, $x, $($v2)*], $n3: [$z, $x, $y, $($v3)*] = $($expr)*);
    };

    (($dx:tt $x:ident, $dy:tt $y:ident, $dz:tt $z:ident) => let $n1:ident, $n2:ident, $n3:ident = $($expr:tt)*) => {
        sep_let!(($dx $x, $dy $y, $dz $z) =>
            let $n1: [$x, $y, $z], $n2: [$y, $z, $x], $n3: [$z, $x, $y] = $($expr)*);
    };
}

// xyzw cyclic permutation shortcut.
// Simplifies cofactor repetition.
macro_rules! sep_xyzw {
    (($dx:tt $x:ident, $dy:tt $y:ident, $dz:tt $z:ident, $dw:tt $w:ident $(, $d:tt $var:ident)*)
        => let $n1:ident : [$($v1:tt)*], $n2:ident : [$($v2:tt)*], $n3:ident : [$($v3:tt)*], $n4:ident : [$($v4:tt)*] = $($expr:tt)*) =>
    {
        sep_let!(($dx $x, $dy $y, $dz $z, $dw $w $(, $d $var)*) =>
            let $n1: [$x, $y, $z, $w, $($v1)*], $n2: [$y, $z, $w, $x, $($v2)*], $n3: [$z, $w, $x, $y, $($v3)*], $n4: [$w, $x, $y, $z, $($v4)*]
                = $($expr)*);
    };

    (($dx:tt $x:ident, $dy:tt $y:ident, $dz:tt $z:ident, $dw:tt $w:ident) => let $n1:ident, $n2:ident, $n3:ident, $n4:ident = $($expr:tt)*) => {
        sep_let!(($dx $x, $dy $y, $dz $z, $dw $w) =>
            let $n1: [$x, $y, $z, $w], $n2: [$y, $z, $w, $x], $n3: [$z, $w, $x, $y], $n4: [$w, $x, $y, $z] = $($expr)*);
    };
}

// xyzw half even permutation shortcut.
// Simplifies cofactor repitition.
macro_rules! sep_xyzw6 {
//    (($dx:tt $x:ident, $dy:tt $y:ident, $dz:tt $z:ident, $dw:tt $w:ident $(, $d:tt $var:ident)*)
//        => let $nxy:ident : [$($vxy:tt)*], $nxz:ident : [$($vxz:tt)*], $nxw:ident : [$($vxw:tt)*]
//               $nyz:ident : [$($vyz:tt)*], $nyw:ident : [$($vyw:tt)*], $nyx:ident : [$($vyx:tt)*]
//               $nzw:ident : [$($vzw:tt)*], $nzx:ident : [$($vzx:tt)*], $nzy:ident : [$($vzy:tt)*]
//               $nwx:ident : [$($vwz:tt)*], $nwy:ident : [$($vwy:tt)*], $nwz:ident : [$($vwz:tt)*]
//        = $($expr:tt)*) =>
//    {
//        sep_let!(($dx $x, $dy $y, $dz $z $(, $d $var)*) =>
//            let $nxy: [$x, $y, $z, $w, $($vxy)*], $nxz: [$x, $z, $w, $y, $($vxz)*], $nxw: [ $($v3)*] = $($expr)*);
//    };

    (($dx:tt $x:ident, $dy:tt $y:ident, $dz:tt $z:ident, $dw:tt $w:ident)
        => let $nxy:ident, $nxz:ident, $nxw:ident, $nyz:ident, $nyw:ident, $nzw:ident = $($expr:tt)*) => {
        sep_let!(($dx $x, $dy $y, $dz $z, $dw $w) =>
            let $nxy: [$x, $y, $z, $w], $nxz: [$x, $z, $w, $y], $nxw: [$x, $w, $y, $z],
                $nyz: [$y, $z, $x, $w], $nyw: [$y, $w, $z, $x], $nzw: [$z, $w, $x, $y],
             = $($expr)*);
    };

    //// Because writing out all 12 terms is annoying
    //(($dx:tt $x:ident, $dy:tt $y:ident, $dz:tt $z:ident, $dw:tt $w:ident)
    //    => let $n:ident = $($expr:tt)*) => {
    //    sep_let!(($dx $x, $dy $y, $dz $z) =>
    //        let paste!([<$n$x$y>]): [$x, $y, $z, $w], paste!([<$n$x$z>]): [$x, $z, $w, $y], paste!([<$n$x$w>]): [$x, $w, $y, $z],
    //            paste!([<$n$y$z>]): [$y, $z, $w, $x], paste!([<$n$y$w>]): [$y, $w, $x, $z], paste!([<$n$y$x>]): [$y, $x, $z, $w],
    //            paste!([<$n$z$w>]): [$z, $w, $x, $y], paste!([<$n$z$x>]): [$z, $x, $y, $w], paste!([<$n$z$y>]): [$z, $y, $w, $x],
    //            paste!([<$n$w$x>]): [$w, $x, $y, $z], paste!([<$n$w$y>]): [$w, $y, $z, $x], paste!([<$n$w$z>]): [$w, $z, $x, $y],
    //         = $($expr)*);
    //};
}

// These regress performance, unfortunately
//macro_rules! arr_let {
//    (($($d:tt $var:ident),*) => let $name:ident : $([$($vals:tt)*]),* = $($expr:tt)*) => {
//        macro_rules! __mac {
//            ($($d $var:ident),*) => { $($expr)* };
//            ($($d $var:expr),*) => { $($expr)* };
//        }
//
//        let $name = [
//            $(__mac!($($vals)*)),*
//        ];
//    };
//}
//
//macro_rules! arr_xyz {
//    (($dx:tt $x:ident, $dy:tt $y:ident, $dz:tt $z:ident) => let $name:ident = $($expr:tt)*) => {
//        arr_let!(($dx $x, $dy $y, $dz $z) =>
//            let $name: [[0, 1, 2], [1, 2, 0], [2, 0, 1]] = $($expr)*);
//    };
//}

/// Calculates the orientation of points `a`, `b`, `c`, `d` in a space.
/// Returns a positive number if `b`→`c`→`d` defines a left turn when looked at from `a`,
/// a negative number if they define a right turn,
/// and 0 if `a`, `b`, `c`, `d` are coplanar.
pub fn orient_3d(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    // vec![[0, 1, 2], [1, 2, 0], [2, 0, 1]].into_iter().map(|[x, y, z] stuff).sum::<f64>() regressed performance a lot
    sep_xyz!(($x, $y, $z) => let cof1, cof2, cof3 = 
        (a.$z - d.$z) * ((b.$x - d.$x) * (c.$y - d.$y) - (b.$y - d.$y) * (c.$x - d.$x)));
    let det = cof1 + cof2 + cof3;

    sep_xyz!(($x, $y, $z) => let cof1_sum, cof2_sum, cof3_sum = 
        (a.$z - d.$z).abs() * (((b.$x - d.$x) * (c.$y - d.$y)).abs() + ((b.$y - d.$y) * (c.$x - d.$x)).abs()));
    let det_sum = cof1_sum + cof2_sum + cof3_sum;

    if det.abs() >= det_sum * ORIENT_3D_BOUND_A {
        det
    } else {
        orient_3d_adapt(a, b, c, d, det_sum)
    }
}

fn orient_3d_adapt(a: Vec3, b: Vec3, c: Vec3, d: Vec3, det_sum: f64) -> f64 {
    sep_xyz!(($x, $y, $z) => let sub1, sub2, sub3 = (two_product(b.$x - d.$x, c.$y - d.$y) - two_product(b.$y - d.$y, c.$x - d.$x)).dynamic());
    sep_xyz!(($x, $y, $z, $s) => let cof1: [sub1], cof2: [sub2], cof3: [sub3] = $s.scale_expansion(a.$z - d.$z));
    let det_hi = cof1 + cof2 + cof3;
    let det_approx = det_hi.approximate();

    if det_approx.abs() >= det_sum * ORIENT_3D_BOUND_B {
        return det_approx;
    }

    // Correction factor for order ε² error bound
    sep_xyz!(($x, $y, $z) => let ax, ay, az = two_sum(a.$x, -d.$x));
    sep_xyz!(($x, $y, $z) => let bx, by, bz = two_sum(b.$x, -d.$x));
    sep_xyz!(($x, $y, $z) => let cx, cy, cz = two_sum(c.$x, -d.$x));
    sep_xyz!(($x, $y, $z) => let cof1_m1, cof2_m1, cof3_m1 =
        paste!([<a$z>][0] * ([<b$x>][1] * [<c$y>][1] - [<b$y>][1] * [<c$x>][1])));
    sep_xyz!(($x, $y, $z) => let cof1_m2, cof2_m2, cof3_m2 =
        paste!([<a$z>][1] * (([<b$x>][0] * [<c$y>][1] + [<b$x>][1] * [<c$y>][0]) - ([<b$y>][0] * [<c$x>][1] + [<b$y>][1] * [<c$x>][0]))));
    let det = cof1_m1 + cof2_m1 + cof3_m1 + cof1_m2 + cof2_m2 + cof3_m2 + det_approx;

    if det.abs() >= ORIENT_3D_BOUND_C1 * det_approx.abs() + ORIENT_3D_BOUND_C2 * det_sum {
        return det;
    }

    // Exact result time!
    let det_m1 =
        sub1.scale_expansion(az[0]) + sub2.scale_expansion(ax[0]) + sub3.scale_expansion(ay[0]);
    sep_xyz!(($x, $y, $z) => let cof1x, cof1y, cof1z =
        paste!((two_product([<b$x>][0], [<c$y>][1]) + two_product([<b$x>][1], [<c$y>][0]) + two_product([<b$x>][0], [<c$y>][0])).dynamic()));
    sep_xyz!(($x, $y, $z) => let cof2x, cof2y, cof2z =
        paste!((two_product([<b$y>][0], [<c$x>][1]) + two_product([<b$y>][1], [<c$x>][0]) + two_product([<b$y>][0], [<c$x>][0])).dynamic()));
    sep_xyz!(($x, $y, $z) => let cof1, cof2, cof3 = paste!([<cof1$x>] - [<cof2$x>]));
    let det_m2 =
        cof1.scale_expansion(az[1]) + cof2.scale_expansion(ax[1]) + cof3.scale_expansion(ay[1]);
    let det_lo =
        cof1.scale_expansion(az[0]) + cof2.scale_expansion(ax[0]) + cof3.scale_expansion(ay[0]);
    let det = (det_hi + det_m1) + (det_m2 + det_lo);
    det.highest_magnitude()
}

/// Returns a positive number if `d` is inside the circle that goes through `a`, `b`, `c`,
/// a negative number if it lies outside,
/// and 0 if `a`, `b`, `c`, `d` are cocircular.
pub fn in_circle(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> f64 {
    sep_xyz!(($a, $b, $c) => let sqa, sqb, sqc = ($a.x - d.x) * ($a.x - d.x) + ($a.y - d.y) * ($a.y - d.y));
    sep_xyz!(($a, $b, $c) => let cof1, cof2, cof3 =
        paste!([<sq$a>]) * (($b.x - d.x) * ($c.y - d.y) - ($b.y - d.y) * ($c.x - d.x)));
    let det = cof1 + cof2 + cof3;

    sep_xyz!(($a, $b, $c) => let cof1_sum, cof2_sum, cof3_sum =
        paste!([<sq$a>]) * ((($b.x - d.x) * ($c.y - d.y)).abs() + (($b.y - d.y) * ($c.x - d.x)).abs()));
    let det_sum = cof1_sum + cof2_sum + cof3_sum;

    if det.abs() >= det_sum * IN_CIRCLE_BOUND_A {
        det
    } else {
        in_circle_adapt(a, b, c, d, det_sum)
    }
}

fn in_circle_adapt(a: Vec2, b: Vec2, c: Vec2, d: Vec2, det_sum: f64) -> f64 {
    sep_xyz!(($a, $b, $c) => let sqa, sqb, sqc = (square($a.x - d.x) + square($a.y - d.y)).dynamic());
    sep_xyz!(($a, $b, $c) => let suba, subb, subc = (two_product($b.x - d.x, $c.y - d.y) - two_product($b.y - d.y, $c.x - d.x)).dynamic());
    sep_xyz!(($a, $b, $c) => let cof1, cof2, cof3 = paste!([<sq$a>] * [<sub$a>]));
    let det = cof1 + cof2 + cof3;
    let det_approx = det.approximate();

    if det_approx.abs() >= det_sum * IN_CIRCLE_BOUND_B {
        return det_approx;
    }

    // Correction factor for order ε² error bound
    sep_xyz!(($a, $b, $c) => let xa, xb, xc = two_sum($a.x, -d.x));
    sep_xyz!(($a, $b, $c) => let ya, yb, yc = two_sum($a.y, -d.y));
    sep_xyz!(($a, $b, $c) => let cof1_m1, cof2_m1, cof3_m1 =
        paste!(([<x$a>][0] * [<x$a>][1] + [<y$a>][0] * [<y$a>][1])
            * ([<x$b>][1] * [<y$c>][1] - [<y$b>][1] * [<x$c>][1])));
    sep_xyz!(($a, $b, $c) => let cof1_m2, cof2_m2, cof3_m2 =
        paste!(([<x$a>][1] * [<x$a>][1] + [<y$a>][1] * [<y$a>][1])
            * (([<x$b>][0] * [<y$c>][1] + [<x$b>][1] * [<y$c>][0]) - ([<y$b>][0] * [<x$c>][1] + [<y$b>][1] * [<x$c>][0]))));
    let det = (cof1_m1 + cof2_m1 + cof3_m1) * 2.0 + cof1_m2 + cof2_m2 + cof3_m2 + det_approx;

    if det.abs() >= IN_CIRCLE_BOUND_C1 * det_approx.abs() + IN_CIRCLE_BOUND_C2 * det_sum {
        return det;
    }

    // Exact result time!
    // Let's be lazy
    sep_xyz!(($a, $b, $c) => let xa, xb, xc = paste!([<x$a>].dynamic()));
    sep_xyz!(($a, $b, $c) => let ya, yb, yc = paste!([<y$a>].dynamic()));
    sep_xyz!(($a, $b, $c) => let sqa, sqb, sqc = paste!(&[<x$a>] * &[<x$a>] + &[<y$a>] * &[<y$a>]));
    sep_xyz!(($a, $b, $c) => let cof1, cof2, cof3 = paste!(&[<sq$a>] * (&[<x$b>] * &[<y$c>] - &[<y$b>] * &[<x$c>])));
    let det = cof1 + cof2 + cof3;
    det.highest_magnitude()
}

/// Returns a positive number if `e` is inside the sphere that goes through `a`, `b`, `c`, `d`,
/// a negative number if it lies outside,
/// and 0 if `a`, `b`, `c`, `d`, `e` are cospherical.
pub fn in_sphere(a: Vec3, b: Vec3, c: Vec3, d: Vec3, e: Vec3) -> f64 {
    // Oh boy. 4x4 determinant, *lots* of cofactors.
    sep_xyzw!(($a, $b, $c, $d) => let sqa, sqb, sqc, sqd =
        ($a.x - e.x) * ($a.x - e.x) + ($a.y - e.y) * ($a.y - e.y) + ($a.z - e.z) * ($a.z - e.z));
    sep_xyzw6!(($a, $b, $c, $d) => let cof1, cof2, cof3, cof4, cof5, cof6 =
        paste!((($a.x - e.x) * ($b.y - e.y) - ($a.y - e.y) * ($b.x - e.x)) *
               (($c.z - e.z) * [<sq$d>] - [<sq$c>] * ($d.z - e.z))));
    let det = cof1 + cof2 + cof3 + cof4 + cof5 + cof6;

    sep_xyzw6!(($a, $b, $c, $d) => let cof1_sum, cof2_sum, cof3_sum, cof4_sum, cof5_sum, cof6_sum =
        paste!(((($a.x - e.x) * ($b.y - e.y)).abs() + (($a.y - e.y) * ($b.x - e.x)).abs()) *
               ((($c.z - e.z) * [<sq$d>]).abs() + ([<sq$c>] * ($d.z - e.z)).abs())));
    let det_sum = cof1_sum + cof2_sum + cof3_sum + cof4_sum + cof5_sum + cof6_sum;

    if det.abs() >= det_sum * IN_SPHERE_BOUND_A {
        det
    } else {
        in_sphere_adapt(a, b, c, d, e, det_sum)
    }
}

fn in_sphere_adapt(a: Vec3, b: Vec3, c: Vec3, d: Vec3, e: Vec3, det_sum: f64) -> f64 {
    sep_xyzw!(($a, $b, $c, $d) => let sqa, sqb, sqc, sqd = (square($a.x - e.x) + square($a.y - e.y)).dynamic() + square($a.z - e.z).dynamic());
    sep_xyzw6!(($a, $b, $c, $d) => let cof1, cof2, cof3, cof4, cof5, cof6 =
        paste!((two_product(($a.x - e.x), ($b.y - e.y)) - two_product(($a.y - e.y), ($b.x - e.x))).dynamic() *
               ([<sq$d>].scale_expansion($c.z - e.z) - [<sq$c>].scale_expansion($d.z - e.z))));
    let det = (cof1 + cof2 + cof3) + (cof4 + cof5 + cof6); // Balanced summation is faster
    let det_approx = det.approximate();

    if det_approx.abs() >= det_sum * IN_SPHERE_BOUND_B {
        return det_approx;
    }

    sep_xyzw!(($a, $b, $c, $d) => let xa, xb, xc, xd = two_sum($a.x, -e.x));
    sep_xyzw!(($a, $b, $c, $d) => let ya, yb, yc, yd = two_sum($a.y, -e.y));
    sep_xyzw!(($a, $b, $c, $d) => let za, zb, zc, zd = two_sum($a.z, -e.z));

    sep_xyzw!(($a, $b, $c, $d) => let sqa, sqb, sqc, sqd =
        paste!([<x$a>][1] * [<x$a>][1] + [<y$a>][1] * [<y$a>][1] + [<z$a>][1] * [<z$a>][1]));
    sep_xyzw!(($a, $b, $c, $d) => let sra, srb, src, srd =
        paste!([<x$a>][0] * [<x$a>][1] + [<y$a>][0] * [<y$a>][1] + [<z$a>][0] * [<z$a>][1]));

    sep_xyzw6!(($a, $b, $c, $d) => let c11, c21, c31, c41, c51, c61 =
        paste!((([<x$a>][0] * [<y$b>][1] + [<x$a>][1] * [<y$b>][0]) - ([<y$a>][0] * [<x$b>][1] + [<y$a>][1] * [<x$b>][0])) *
               ([<z$c>][1] * [<sq$d>] - [<sq$c>] * [<z$d>][1])));
    sep_xyzw6!(($a, $b, $c, $d) => let c12, c22, c32, c42, c52, c62 =
        paste!(([<x$a>][1] * [<y$b>][1] - [<y$a>][1] - [<x$b>][1]) *
               (([<z$c>][0] * [<sq$d>] + [<z$c>][1] * [<sr$d>]) - ([<sr$c>] * [<z$d>][1] + [<sq$c>] * [<z$d>][0]))));
    let det =
        c11 + c21 + c31 + c41 + c51 + c61 + (c12 + c22 + c32 + c42 + c52 + c62) * 2.0 + det_approx;

    if det.abs() >= IN_SPHERE_BOUND_C1 * det_approx.abs() + IN_SPHERE_BOUND_C2 * det_sum {
        return det;
    }

    // Exact result time!
    // Let's be lazy
    sep_xyzw!(($a, $b, $c, $d) => let xa, xb, xc, xd = paste!([<x$a>].dynamic()));
    sep_xyzw!(($a, $b, $c, $d) => let ya, yb, yc, yd = paste!([<y$a>].dynamic()));
    sep_xyzw!(($a, $b, $c, $d) => let za, zb, zc, zd = paste!([<z$a>].dynamic()));
    sep_xyzw!(($a, $b, $c, $d) => let sqa, sqb, sqc, sqd =
        paste!(&[<x$a>] * &[<x$a>] + &[<y$a>] * &[<y$a>] + &[<z$a>] * &[<z$a>]));
    sep_xyzw6!(($a, $b, $c, $d) => let cof1, cof2, cof3, cof4, cof5, cof6 =
        paste!((&[<x$a>] * &[<y$b>] - &[<y$a>] * &[<x$b>]) * (&[<z$c>] * &[<sq$d>] - &[<sq$c>] * &[<z$d>])));
    let det = (cof1 + cof2 + cof3) + (cof4 + cof5 + cof6);
    det.highest_magnitude()
}

#[cfg(test)]
mod test {
    use super::*;
    use nalgebra::Matrix2;
    use rand::distributions::{Distribution, Uniform};
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rand_distr::UnitCircle;
    use rand_pcg::Pcg64;
    use rug::Float;

    type Mtx2 = Matrix2<f64>;

    const PCG_STATE: u128 = 0xcafef00dd15ea5e5;
    const PCG_STREAM: u128 = 0xa02bdbf7bb3c0a7ac28fa16a64abf96;

    fn orient_2d_exact(a: Vec2, b: Vec2, c: Vec2) -> Float {
        const PREC: u32 = (f64::MANTISSA_DIGITS + 1) * 2 + 1;
        let ax = Float::with_val(
            PREC,
            &Float::with_val(PREC, a.x) - &Float::with_val(PREC, c.x),
        );
        let ay = Float::with_val(
            PREC,
            &Float::with_val(PREC, a.y) - &Float::with_val(PREC, c.y),
        );
        let bx = Float::with_val(
            PREC,
            &Float::with_val(PREC, b.x) - &Float::with_val(PREC, c.x),
        );
        let by = Float::with_val(
            PREC,
            &Float::with_val(PREC, b.y) - &Float::with_val(PREC, c.y),
        );
        Float::with_val(PREC, &ax * &by - &ay * &bx)
    }

    fn check_orient_2d(a: Vec2, b: Vec2, c: Vec2) {
        let adapt = orient_2d(a, b, c);
        let exact = orient_2d_exact(a, b, c);
        assert_eq!(
            adapt.partial_cmp(&0.0),
            exact.partial_cmp(&0.0),
            "({}, {}, {}) gave wrong result: {} vs {}",
            a,
            b,
            c,
            adapt,
            exact
        );
    }

    #[test]
    fn test_orient_2d_uniform_random() {
        // Deterministic, portable RNG
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new(-10.0, 10.0);

        for _ in 0..10000 {
            let vals = dist.sample_iter(&mut rng).take(6).collect::<Vec<_>>();
            let a = Vec2::new(vals[0], vals[1]);
            let b = Vec2::new(vals[2], vals[3]);
            let c = Vec2::new(vals[4], vals[5]);
            check_orient_2d(a, b, c);
        }
    }

    #[test]
    fn test_orient_2d_geometric_random() {
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let mut rng2 = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new(-30.0, 30.0);

        for _ in 0..10000 {
            let vals = dist
                .sample_iter(&mut rng)
                .take(6)
                .map(|x: f64| if rng2.gen() { -1.0 } else { 1.0 } * x.exp2())
                .collect::<Vec<_>>();
            let a = Vec2::new(vals[0], vals[1]);
            let b = Vec2::new(vals[2], vals[3]);
            let c = Vec2::new(vals[4], vals[5]);
            check_orient_2d(a, b, c);
        }
    }

    #[test]
    fn test_orient_2d_near_collinear() {
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new(-1.0, 1.0);
        let fac_dist = Uniform::new(-1.0, 2.0);

        for _ in 0..10000 {
            let vals = dist.sample_iter(&mut rng).take(4).collect::<Vec<_>>();
            let a = Vec2::new(vals[0], vals[1]);
            let b = Vec2::new(vals[2], vals[3]);

            let fac = fac_dist.sample(&mut rng);
            let c = a + (b - a) * fac;

            check_orient_2d(a, b, c);
        }
    }

    #[test]
    fn test_orient_2d_collinear() {
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new_inclusive(-4096, 4096);
        let fac_dist = Uniform::new_inclusive(-4095, 4096);

        for _ in 0..10000 {
            let vals = dist
                .sample_iter(&mut rng)
                .take(4)
                .map(|x| (x as f64) / 4096.0)
                .collect::<Vec<_>>();
            let a = Vec2::new(vals[0], vals[1]);
            let b = Vec2::new(vals[2], vals[3]);

            let fac = fac_dist.sample(&mut rng);
            let c = a + (b - a) * (fac as f64) / 16.0;

            check_orient_2d(a, b, c);
        }
    }

    fn orient_3d_exact(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Float {
        const PREC: u32 = (f64::MANTISSA_DIGITS + 1) * 3 + 3;
        let ax = Float::with_val(
            PREC,
            &Float::with_val(PREC, a.x) - &Float::with_val(PREC, d.x),
        );
        let ay = Float::with_val(
            PREC,
            &Float::with_val(PREC, a.y) - &Float::with_val(PREC, d.y),
        );
        let az = Float::with_val(
            PREC,
            &Float::with_val(PREC, a.z) - &Float::with_val(PREC, d.z),
        );
        let bx = Float::with_val(
            PREC,
            &Float::with_val(PREC, b.x) - &Float::with_val(PREC, d.x),
        );
        let by = Float::with_val(
            PREC,
            &Float::with_val(PREC, b.y) - &Float::with_val(PREC, d.y),
        );
        let bz = Float::with_val(
            PREC,
            &Float::with_val(PREC, b.z) - &Float::with_val(PREC, d.z),
        );
        let cx = Float::with_val(
            PREC,
            &Float::with_val(PREC, c.x) - &Float::with_val(PREC, d.x),
        );
        let cy = Float::with_val(
            PREC,
            &Float::with_val(PREC, c.y) - &Float::with_val(PREC, d.y),
        );
        let cz = Float::with_val(
            PREC,
            &Float::with_val(PREC, c.z) - &Float::with_val(PREC, d.z),
        );
        let xy = Float::with_val(PREC, &bx * &cy - &by * &cx);
        let yz = Float::with_val(PREC, &by * &cz - &bz * &cy);
        let zx = Float::with_val(PREC, &bz * &cx - &bx * &cz);
        let ab = Float::with_val(PREC, &az * &xy + &ax * &yz);
        Float::with_val(PREC, &ab + &ay * &zx)
    }

    fn check_orient_3d(a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        let adapt = orient_3d(a, b, c, d);
        let exact = orient_3d_exact(a, b, c, d);
        assert_eq!(
            adapt.partial_cmp(&0.0),
            exact.partial_cmp(&0.0),
            "({}, {}, {}, {}) gave wrong result: {} vs {}",
            a,
            b,
            c,
            d,
            adapt,
            exact
        );
    }

    #[test]
    fn test_orient_3d_uniform_random() {
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new(-10.0, 10.0);

        for _ in 0..1000 {
            let vals = dist.sample_iter(&mut rng).take(12).collect::<Vec<_>>();
            let a = Vec3::new(vals[0], vals[1], vals[2]);
            let b = Vec3::new(vals[3], vals[4], vals[5]);
            let c = Vec3::new(vals[6], vals[7], vals[8]);
            let d = Vec3::new(vals[9], vals[10], vals[11]);
            check_orient_3d(a, b, c, d);
        }
    }

    #[test]
    fn test_orient_3d_geometric_random() {
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let mut rng2 = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new(-30.0, 30.0);

        for _ in 0..1000 {
            let vals = dist
                .sample_iter(&mut rng)
                .take(12)
                .map(|x: f64| if rng2.gen() { -1.0 } else { 1.0 } * x.exp2())
                .collect::<Vec<_>>();
            let a = Vec3::new(vals[0], vals[1], vals[2]);
            let b = Vec3::new(vals[3], vals[4], vals[5]);
            let c = Vec3::new(vals[6], vals[7], vals[8]);
            let d = Vec3::new(vals[9], vals[10], vals[11]);
            check_orient_3d(a, b, c, d);
        }
    }

    #[test]
    fn test_orient_3d_near_coplanar() {
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new(-1.0, 1.0);
        let fac_dist = Uniform::new(-1.0, 2.0);

        for _ in 0..1000 {
            let vals = dist.sample_iter(&mut rng).take(9).collect::<Vec<_>>();
            let a = Vec3::new(vals[0], vals[1], vals[2]);
            let b = Vec3::new(vals[3], vals[4], vals[5]);
            let c = Vec3::new(vals[6], vals[7], vals[8]);

            let fac = fac_dist.sample(&mut rng);
            let fac2 = fac_dist.sample(&mut rng);
            let d = c + (a + (b - a) * fac - c) * fac2;

            check_orient_3d(a, b, c, d);
        }
    }

    #[test]
    fn test_orient_3d_coplanar() {
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new_inclusive(-4096, 4096);
        let fac_dist = Uniform::new_inclusive(-4095, 4096);

        for _ in 0..1000 {
            let vals = dist
                .sample_iter(&mut rng)
                .take(9)
                .map(|x| (x as f64) / 4096.0)
                .collect::<Vec<_>>();
            let a = Vec3::new(vals[0], vals[1], vals[2]);
            let b = Vec3::new(vals[3], vals[4], vals[5]);
            let c = Vec3::new(vals[6], vals[7], vals[8]);

            let fac = fac_dist.sample(&mut rng) as f64 / 16.0;
            let fac2 = fac_dist.sample(&mut rng) as f64 / 16.0;
            let d = c + (a + (b - a) * fac - c) * fac2;

            check_orient_3d(a, b, c, d);
        }
    }

    fn in_circle_exact(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> Float {
        const PREC: u32 = ((f64::MANTISSA_DIGITS + 1) * 2 + 1) * 2 + 2;
        macro_rules! f {
            ($expr:expr) => {
                Float::with_val(PREC, $expr)
            };
        }

        let ax = f!(&f!(a.x) - &f!(d.x));
        let ay = f!(&f!(a.y) - &f!(d.y));
        let bx = f!(&f!(b.x) - &f!(d.x));
        let by = f!(&f!(b.y) - &f!(d.y));
        let cx = f!(&f!(c.x) - &f!(d.x));
        let cy = f!(&f!(c.y) - &f!(d.y));
        let cof1 = f!(&f!(&ax * &ax + &ay * &ay) * &f!(&bx * &cy - &by * &cx));
        let cof2 = f!(&f!(&bx * &bx + &by * &by) * &f!(&cx * &ay - &cy * &ax));
        let cof3 = f!(&f!(&cx * &cx + &cy * &cy) * &f!(&ax * &by - &ay * &bx));
        f!(&f!(&cof1 + &cof2) + &cof3)
    }

    fn check_in_circle(a: Vec2, b: Vec2, c: Vec2, d: Vec2) {
        let adapt = in_circle(a, b, c, d);
        let exact = in_circle_exact(a, b, c, d);
        assert_eq!(
            adapt.partial_cmp(&0.0),
            exact.partial_cmp(&0.0),
            "({}, {}, {}, {}) gave wrong result: {} vs {}",
            a,
            b,
            c,
            d,
            adapt,
            exact
        );
    }

    #[test]
    fn test_in_circle_uniform_random() {
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new(-10.0, 10.0);

        for _ in 0..1000 {
            let vals = dist.sample_iter(&mut rng).take(8).collect::<Vec<_>>();
            let a = Vec2::new(vals[0], vals[1]);
            let b = Vec2::new(vals[2], vals[3]);
            let c = Vec2::new(vals[4], vals[5]);
            let d = Vec2::new(vals[6], vals[7]);
            check_in_circle(a, b, c, d);
        }
    }

    #[test]
    fn test_in_circle_geometric_random() {
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let mut rng2 = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new(-30.0, 30.0);

        for _ in 0..1000 {
            let vals = dist
                .sample_iter(&mut rng)
                .take(12)
                .map(|x: f64| if rng2.gen() { -1.0 } else { 1.0 } * x.exp2())
                .collect::<Vec<_>>();
            let a = Vec2::new(vals[0], vals[1]);
            let b = Vec2::new(vals[2], vals[3]);
            let c = Vec2::new(vals[4], vals[5]);
            let d = Vec2::new(vals[6], vals[7]);
            check_in_circle(a, b, c, d);
        }
    }

    #[test]
    fn test_in_circle_near_cocircular() {
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new(-1.0, 1.0);

        for _ in 0..1000 {
            let vals = UnitCircle.sample_iter(&mut rng).take(4).collect::<Vec<_>>();
            let radius = dist.sample(&mut rng);
            let x = dist.sample(&mut rng);
            let y = dist.sample(&mut rng);
            let offset = Vec2::new(x, y);

            let a = Vec2::new(vals[0][0], vals[0][1]) * radius + offset;
            let b = Vec2::new(vals[1][0], vals[1][1]) * radius + offset;
            let c = Vec2::new(vals[2][0], vals[2][1]) * radius + offset;
            let d = Vec2::new(vals[3][0], vals[3][1]) * radius + offset;

            check_in_circle(a, b, c, d);
        }
    }

    #[test]
    fn test_in_circle_cocircular() {
        let mut rng = Pcg64::new(PCG_STATE, PCG_STREAM);
        let dist = Uniform::new_inclusive(-4096, 4096);
        let abs = Uniform::new_inclusive(0, 4096);
        let mtxs = [
            Mtx2::new(1.0, 0.0, 0.0, 1.0),
            Mtx2::new(-1.0, 0.0, 0.0, 1.0),
            Mtx2::new(0.0, -1.0, 1.0, 0.0),
            Mtx2::new(0.0, 1.0, 1.0, 0.0),
            Mtx2::new(-1.0, 0.0, 0.0, -1.0),
            Mtx2::new(1.0, 0.0, 0.0, -1.0),
            Mtx2::new(0.0, 1.0, -1.0, 0.0),
            Mtx2::new(0.0, -1.0, -1.0, 0.0),
        ];

        for _ in 0..1000 {
            let x = abs.sample(&mut rng) as f64 / 4096.0;
            let y = abs.sample(&mut rng) as f64 / 4096.0;
            let v = Vec2::new(x, y);
            let cx = dist.sample(&mut rng) as f64 / 4096.0;
            let cy = dist.sample(&mut rng) as f64 / 4096.0;
            let offset = Vec2::new(cx, cy);

            // Reminder to use choose_multiple for the benchmark to not mix
            // cases that have identical points and are thus more likely to
            // get stopped at the initial floating-point calculation
            let a = mtxs.choose(&mut rng).unwrap() * v + offset;
            let b = mtxs.choose(&mut rng).unwrap() * v + offset;
            let c = mtxs.choose(&mut rng).unwrap() * v + offset;
            let d = mtxs.choose(&mut rng).unwrap() * v + offset;

            check_in_circle(a, b, c, d);
        }
    }
}