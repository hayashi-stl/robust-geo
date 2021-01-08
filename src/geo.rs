//! Contains the geometric tests

use crate::{two_product, two_sum};
use crate::Expansion;

use nalgebra::Vector2;
type Vec2 = Vector2<f64>;

const EPSILON: f64 = f64::EPSILON / 2.0;

const ORIENT_2D_BOUND_A: f64 = (3.0 + 16.0 * EPSILON) * EPSILON;
const ORIENT_2D_BOUND_B: f64 = (2.0 + 12.0 * EPSILON) * EPSILON;
const ORIENT_2D_BOUND_C1: f64 = (3.0 + 8.0 * EPSILON) * EPSILON;
const ORIENT_2D_BOUND_C2: f64 = (9.0 + 64.0 * EPSILON) * EPSILON * EPSILON;

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
        let dist = Uniform::new(-2.0, 2.0);
        let fac_dist = Uniform::new(-1.0, 2.0);
        let perturb = Uniform::new(-(-50f64).exp2(), (-50f64).exp2());

        for _ in 0..10000 {
            let vals = dist.sample_iter(&mut rng).take(4).collect::<Vec<_>>();
            let a = Vec2::new(vals[0], vals[1]);
            let b = Vec2::new(vals[2], vals[3]);

            let fac = fac_dist.sample(&mut rng);
            let pert = perturb.sample_iter(&mut rng).take(2).collect::<Vec<_>>();
            let c = a + (b - a) * fac + Vec2::new(pert[0], pert[1]);
            
            check_orient_2d(a, b, c);
        }
    }
}