//! Contains the geometric tests

use crate::{two_product, two_sum};
use crate::Expansion;

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

/// Calculates the orientation of points `a`, `b`, `c` in a plane.
/// Returns a positive number if they define a left turn,
/// a negative number if they define a right turn,
/// and 0 if they are collinear.
#[rustfmt::skip]
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

#[rustfmt::skip]
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
    (($($d:tt $var:ident),*) => let $($name:ident : [$($vals:tt)*]),* = $($expr:tt)*) => {
        macro_rules! __mac {
            ($($d $var:ident),*) => { $($expr)* };
            ($($d $var:expr),*) => { $($expr)* };
        }

        let [$($name),*] = [
            $(__mac!($($vals)*)),*
        ];
    };
}

// xyz permutation shortcut.
// Simplifies cofactor repitition.
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
#[rustfmt::skip]
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

#[rustfmt::skip]
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

    if det.abs() >= ORIENT_3D_BOUND_C1 * det_approx + ORIENT_3D_BOUND_C2 * det_sum {
        return det;
    }

    // Exact result time!
    let det_m1 = sub1.scale_expansion(az[0]) + sub2.scale_expansion(ax[0]) + sub3.scale_expansion(ay[0]);
    sep_xyz!(($x, $y, $z) => let cof1x, cof1y, cof1z =
        paste!((two_product([<b$x>][0], [<c$y>][1]) + two_product([<b$x>][1], [<c$y>][0]) + two_product([<b$x>][0], [<c$y>][0])).dynamic()));
    sep_xyz!(($x, $y, $z) => let cof2x, cof2y, cof2z =
        paste!((two_product([<b$y>][0], [<c$x>][1]) + two_product([<b$y>][1], [<c$x>][0]) + two_product([<b$y>][0], [<c$x>][0])).dynamic()));
    sep_xyz!(($x, $y, $z) => let cof1, cof2, cof3 = paste!([<cof1$x>] - [<cof2$x>]));
    let det_m2 = cof1.scale_expansion(az[1]) + cof2.scale_expansion(ax[1]) + cof3.scale_expansion(ay[1]);
    let det_lo = cof1.scale_expansion(az[0]) + cof2.scale_expansion(ax[0]) + cof3.scale_expansion(ay[0]);
    let det = (det_hi + det_m1) + (det_m2 + det_lo);
    det.highest_magnitude()
}

/// Calculates the orientation of points `a`, `b`, `c`, `d` in a space.
/// Returns a positive number if `b`→`c`→`d` defines a left turn when looked at from `a`,
/// a negative number if they define a right turn,
/// and 0 if `a`, `b`, `c`, `d` are coplanar.
#[rustfmt::skip]
pub fn in_circle_2d(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> f64 {
    todo!()
}

#[cfg(test)]
mod test {
    use super::*;
    use rug::Float;
    use rand::Rng;
    use rand::distributions::{Uniform, Distribution};
    use rand_pcg::Pcg64;

    const PCG_STATE: u128 = 0xcafef00dd15ea5e5;
    const PCG_STREAM: u128 = 0xa02bdbf7bb3c0a7ac28fa16a64abf96;

    fn orient_2d_exact(a: Vec2, b: Vec2, c: Vec2) -> Float {
        const PREC: u32 = (f64::MANTISSA_DIGITS + 1) * 2 + 1;
        let ax = Float::with_val(PREC, &Float::with_val(PREC, a.x) - &Float::with_val(PREC, c.x));
        let ay = Float::with_val(PREC, &Float::with_val(PREC, a.y) - &Float::with_val(PREC, c.y));
        let bx = Float::with_val(PREC, &Float::with_val(PREC, b.x) - &Float::with_val(PREC, c.x));
        let by = Float::with_val(PREC, &Float::with_val(PREC, b.y) - &Float::with_val(PREC, c.y));
        Float::with_val(PREC, &ax * &by - &ay * &bx)
    }

    fn check_orient_2d(a: Vec2, b: Vec2, c: Vec2) {
        let adapt = orient_2d(a, b, c); 
        let exact = orient_2d_exact(a, b, c); 
        assert_eq!(adapt.partial_cmp(&0.0), exact.partial_cmp(&0.0),
            "({}, {}, {}) gave wrong result: {} vs {}", a, b, c, adapt, exact);
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
            let vals = dist.sample_iter(&mut rng).take(6)
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
            let vals = dist.sample_iter(&mut rng).take(4)
                .map(|x| (x as f64) / 4096.0).collect::<Vec<_>>();
            let a = Vec2::new(vals[0], vals[1]);
            let b = Vec2::new(vals[2], vals[3]);

            let fac = fac_dist.sample(&mut rng);
            let c = a + (b - a) * (fac as f64) / 16.0;
            
            check_orient_2d(a, b, c);
        }
    }

    fn orient_3d_exact(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Float {
        const PREC: u32 = (f64::MANTISSA_DIGITS + 1) * 3 + 3;
        let ax = Float::with_val(PREC, &Float::with_val(PREC, a.x) - &Float::with_val(PREC, d.x));
        let ay = Float::with_val(PREC, &Float::with_val(PREC, a.y) - &Float::with_val(PREC, d.y));
        let az = Float::with_val(PREC, &Float::with_val(PREC, a.z) - &Float::with_val(PREC, d.z));
        let bx = Float::with_val(PREC, &Float::with_val(PREC, b.x) - &Float::with_val(PREC, d.x));
        let by = Float::with_val(PREC, &Float::with_val(PREC, b.y) - &Float::with_val(PREC, d.y));
        let bz = Float::with_val(PREC, &Float::with_val(PREC, b.z) - &Float::with_val(PREC, d.z));
        let cx = Float::with_val(PREC, &Float::with_val(PREC, c.x) - &Float::with_val(PREC, d.x));
        let cy = Float::with_val(PREC, &Float::with_val(PREC, c.y) - &Float::with_val(PREC, d.y));
        let cz = Float::with_val(PREC, &Float::with_val(PREC, c.z) - &Float::with_val(PREC, d.z));
        let xy = Float::with_val(PREC, &bx * &cy - &by * &cx);
        let yz = Float::with_val(PREC, &by * &cz - &bz * &cy);
        let zx = Float::with_val(PREC, &bz * &cx - &bx * &cz);
        let ab = Float::with_val(PREC, &az * &xy + &ax * &yz);
        Float::with_val(PREC, &ab + &ay * &zx)
    }

    fn check_orient_3d(a: Vec3, b: Vec3, c: Vec3, d: Vec3) {
        let adapt = orient_3d(a, b, c, d); 
        let exact = orient_3d_exact(a, b, c, d); 
        assert_eq!(adapt.partial_cmp(&0.0), exact.partial_cmp(&0.0),
            "({}, {}, {}, {}) gave wrong result: {} vs {}", a, b, c, d, adapt, exact);
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
            let vals = dist.sample_iter(&mut rng).take(12)
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
            let vals = dist.sample_iter(&mut rng).take(9)
                .map(|x| (x as f64) / 4096.0).collect::<Vec<_>>();
            let a = Vec3::new(vals[0], vals[1], vals[2]);
            let b = Vec3::new(vals[3], vals[4], vals[5]);
            let c = Vec3::new(vals[6], vals[7], vals[8]);

            let fac = fac_dist.sample(&mut rng) as f64 / 16.0;
            let fac2 = fac_dist.sample(&mut rng) as f64 / 16.0;
            let d = c + (a + (b - a) * fac - c) * fac2;
            
            check_orient_3d(a, b, c, d);
        }
    }

}