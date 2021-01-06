//! Implementation of the floating point expansions found in
//! [Adaptive Precision Floating-Point Arithmetic and
//! Fast Robust Geometric Predicates](https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf)
//! in a more Rusty style.
//! Double precision only.

extern crate typenum;
#[macro_use]
extern crate generic_array;

use std::marker::PhantomData;
use typenum::{Greater, Unsigned, U2};
use generic_array::{GenericArray, ArrayLength};

/// Consecutive terms in the expansion do not overlap
/// even if the smaller term is multiplied by 2.
/// Nearly all operations preserve this,
/// assuming the rounding mode is round nearest, halfway to even
trait Nonadjacent {}

/// Adjacent terms are powers of 2 and
/// no term is adjacent to 2 others.
/// Notably preserved by `fast_expansion_sum`,
/// which preserves neither nonadjacency nor nonoverlapping.
trait StronglyNonoverlapping {}

/// Describes the overlap-related property that an expansion has
trait Property {}

struct NAdj;
impl Property for NAdj {}
impl Nonadjacent for NAdj {}
impl StronglyNonoverlapping for NAdj {}

struct SNOver;
impl Property for SNOver {}
impl StronglyNonoverlapping for SNOver {}

struct NOver;
impl Property for NOver {}

/// The trait representing a generic floating-point expansion.
/// The expansion is assumed to be nonoverlapping and
/// terms are in increasing-magnitude order.
trait Expansion {
    /// The number of terms in the expansion.
    fn len(&self) -> usize;
}

/// An expansion whose size is its array size
#[derive(Clone, Debug)]
struct FixedExpansion<P: Property, N: ArrayLength<f64>> {
    arr: GenericArray<f64, N>,
    marker: PhantomData<P>,
}

impl<P: Property, N: ArrayLength<f64>> Expansion for FixedExpansion<P, N> {
    fn len(&self) -> usize {
        N::USIZE
    }
}

impl<P: Property, N: ArrayLength<f64>> FixedExpansion<P, N> {
    fn new(arr: impl Into<GenericArray<f64, N>>) -> Self {
        Self {
            arr: arr.into(),
            marker: PhantomData,
        }
    }
}

/// Constructs an expansion from the sum of `a` and `b`,
/// assuming that |`a`| is at least as big as |`b`|.
/// The greater term is `a` + `b` with floating-point addition.
fn fast_two_sum(a: f64, b: f64) -> FixedExpansion<NAdj, U2> {
    let x = a + b;
    FixedExpansion::new([b - (x - a), x])
}

/// Constructs an expansion from the sum of `a` and `b`.
/// The greater term is `a` + `b` with floating-point addition.
fn two_sum(a: f64, b: f64) -> FixedExpansion<NAdj, U2> {
    let x = a + b;
    let bv = x - a;
    let av = x - bv;
    FixedExpansion::new([(a - av) + (b - bv), x])
}

/// An expansion whose size is stored explicitly,
/// but (for now) still has an upper bound for size.
#[derive(Clone, Debug)]
struct DynamicExpansion<P: Property, N: ArrayLength<f64>> {
    arr: GenericArray<f64, N>,
    len: usize,
    marker: PhantomData<P>,
}

impl<P: Property, N: ArrayLength<f64>> Expansion for DynamicExpansion<P, N> {
    fn len(&self) -> usize {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(sig: f64, exp: i32) -> f64 {
        sig * 2f64.powi(exp)
    }

    #[test]
    fn fast_two_sum_low_precision() {
        // Result can fit in f64
        let nums = [15.75, 1.5, 0.0, -4.625, -15.75];
        for a in &nums {
            for b in &nums {
                if a >= b {
                    let res = fast_two_sum(*a, *b);
                    assert_eq!(res.arr, [0.0, a + b].into());
                }
            }
        }
    }

    #[test]
    fn fast_two_sum_high_precision() {
        // Result is just [b, a]
        let nums_a = [15.75, 1.5, -4.625, -15.75];
        let nums_b = [e(15.75, -70), e(1.5, -70), e(-4.625, -70), e(-15.75, -70)];
        for a in &nums_a {
            for b in &nums_b {
                let res = fast_two_sum(*a, *b);
                assert_eq!(res.arr, [*b, *a].into());
            }
        }
    }

    #[test]
    fn fast_two_sum_split_b() {
        // b is split in result
        let res = fast_two_sum(1.0, e(11.0, -54));
        assert_eq!(res.arr, [e(-1.0, -54), 1.0 + e(3.0, -52)].into());

        let res = fast_two_sum(1.0, e(15.0, -54));
        assert_eq!(res.arr, [e(-1.0, -54), 1.0 + e(1.0, -50)].into());

        let res = fast_two_sum(1.0, -e(11.0, -54));
        assert_eq!(res.arr, [e(1.0, -54), 1.0 - e(3.0, -52)].into());
    }

    #[test]
    fn two_sum_low_precision() {
        // Result can fit in f64
        let nums = [15.75, 1.5, 0.0, -4.625, -15.75];
        for a in &nums {
            for b in &nums {
                let res = two_sum(*a, *b);
                assert_eq!(res.arr, [0.0, a + b].into());
            }
        }
    }

    #[test]
    fn two_sum_high_precision() {
        // Result is just [b, a]
        let nums_a = [15.75, 1.5, -4.625, -15.75];
        let nums_b = [15.75 * 2f64.powi(-70), 1.5 * 2f64.powi(-70), -4.625 * 2f64.powi(-70), -15.75 * 2f64.powi(-70)];
        for a in &nums_a {
            for b in &nums_b {
                let res = two_sum(*a, *b);
                assert_eq!(res.arr, [*b, *a].into());
            }
        }
        for a in &nums_a {
            for b in &nums_b {
                let res = two_sum(*b, *a);
                assert_eq!(res.arr, [*b, *a].into());
            }
        }
    }

    #[test]
    fn two_sum_split_b() {
        // b is split in result
        let res = two_sum(1.0, e(11.0, -54));
        assert_eq!(res.arr, [e(-1.0, -54), 1.0 + e(3.0, -52)].into());
        let res = two_sum(e(11.0, -54), 1.0);
        assert_eq!(res.arr, [e(-1.0, -54), 1.0 + e(3.0, -52)].into());

        let res = two_sum(1.0, e(15.0, -54));
        assert_eq!(res.arr, [e(-1.0, -54), 1.0 + e(1.0, -50)].into());
        let res = two_sum(e(15.0, -54), 1.0);
        assert_eq!(res.arr, [e(-1.0, -54), 1.0 + e(1.0, -50)].into());

        let res = two_sum(1.0, -e(11.0, -54));
        assert_eq!(res.arr, [e(1.0, -54), 1.0 - e(3.0, -52)].into());
        let res = two_sum(-e(11.0, -54), 1.0);
        assert_eq!(res.arr, [e(1.0, -54), 1.0 - e(3.0, -52)].into());
    }
}