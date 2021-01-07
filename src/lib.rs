//! Implementation of the floating point expansions found in
//! [Adaptive Precision Floating-Point Arithmetic and
//! Fast Robust Geometric Predicates](https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf)
//! in a more Rusty style.
//! Double precision only.

extern crate typenum;
#[macro_use]
extern crate generic_array;

use generic_array::{ArrayLength, GenericArray};
use std::marker::PhantomData;
use std::ops::Add;
use typenum::{Add1, Greater, Unsigned, U0, U1, U2};

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
trait Property: Default {
    /// Does not preserve strongly nonoverlapping unless nonadjacent
    type Weak: Property;
}

/// Takes the weaker of two properties
trait Min<P: Property> {
    type Output: Property;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NAdj;
impl Property for NAdj {
    type Weak = NAdj;
}
impl Nonadjacent for NAdj {}
impl StronglyNonoverlapping for NAdj {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SNOver;
impl Property for SNOver {
    type Weak = NOver;
}
impl StronglyNonoverlapping for SNOver {}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NOver;
impl Property for NOver {
    type Weak = NOver;
}

impl Min<NAdj> for NAdj {
    type Output = NAdj;
}
impl Min<NAdj> for SNOver {
    type Output = SNOver;
}
impl Min<NAdj> for NOver {
    type Output = NOver;
}
impl Min<SNOver> for NAdj {
    type Output = SNOver;
}
impl Min<SNOver> for SNOver {
    type Output = SNOver;
}
impl Min<SNOver> for NOver {
    type Output = NOver;
}
impl Min<NOver> for NAdj {
    type Output = NOver;
}
impl Min<NOver> for SNOver {
    type Output = NOver;
}
impl Min<NOver> for NOver {
    type Output = NOver;
}

trait ExpansionKind<P: Property, N: ArrayLength<f64>> {
    type Type: Expansion;
}

/// The trait representing a generic floating-point expansion.
/// The expansion is assumed to be nonoverlapping and
/// terms are in increasing-magnitude order.
trait Expansion {
    type Prop: Property;
    type Bound: ArrayLength<f64>;
    /// Whether to remove zero-terms from expansion
    /// during certain operations
    const ZERO_ELIMINATE: bool;

    fn arr(&self) -> &GenericArray<f64, Self::Bound>;

    /// The number of terms in the expansion.
    fn len(&self) -> usize;

    /// Creates an expansion with a specific length set.
    fn with_len(arr: impl Into<GenericArray<f64, Self::Bound>>, len: usize) -> Self;

    /// Adds `b` to this expansion,
    /// using a sequence of `two_sum`s.
    fn grow_expansion(&self, b: f64) -> Self::Type
    where
        Self: ExpansionKind<
            <<Self as Expansion>::Prop as Property>::Weak,
            <<Self as Expansion>::Bound as Add<U1>>::Output,
        >,
        <Self as Expansion>::Bound: Add<U1>,
        <<Self as Expansion>::Bound as Add<U1>>::Output: ArrayLength<f64>,
    {
        let mut arr = GenericArray::<f64, <Self::Type as Expansion>::Bound>::default();
        let mut sum = b;
        let mut arr_i = 0;

        for i in 0..self.len() {
            let res = two_sum(sum, self.arr()[i]);
            sum = res.arr[1];
            if !Self::ZERO_ELIMINATE || res.arr[0] != 0.0 {
                arr[arr_i] = res.arr[0];
                arr_i += 1;
            }
        }

        if !Self::ZERO_ELIMINATE || sum != 0.0 {
            arr[arr_i] = sum;
            arr_i += 1;
        }

        Self::Type::with_len(arr, arr_i)
    }

}

/// An expansion whose size is its array size
#[derive(Clone, Debug, Default)]
struct FixedExpansion<P: Property, N: ArrayLength<f64>> {
    arr: GenericArray<f64, N>,
    marker: PhantomData<P>,
}

impl<P1: Property, P2: Property, N1: ArrayLength<f64>, N2: ArrayLength<f64>> ExpansionKind<P2, N2>
    for FixedExpansion<P1, N1>
{
    type Type = FixedExpansion<P2, N2>;
}

impl<P: Property, N: ArrayLength<f64>> Expansion for FixedExpansion<P, N> {
    type Prop = P;
    type Bound = N;
    const ZERO_ELIMINATE: bool = false;

    fn arr(&self) -> &GenericArray<f64, Self::Bound> {
        &self.arr
    }

    fn len(&self) -> usize {
        N::USIZE
    }

    fn with_len(arr: impl Into<GenericArray<f64, N>>, len: usize) -> Self {
        debug_assert_eq!(len, N::USIZE);
        Self::new(arr)
    }
}

impl<P: Property, N: ArrayLength<f64>> FixedExpansion<P, N> {
    fn new(arr: impl Into<GenericArray<f64, N>>) -> Self {
        Self {
            arr: arr.into(),
            marker: PhantomData,
        }
    }

    fn subexpansion<Len: ArrayLength<f64>>(&self, start: usize) ->
        FixedExpansion<P, Len>
    {
        let mut arr = GenericArray::<f64, Len>::default();
        for i in 0..Len::USIZE {
            arr[i] = self.arr[start + i];
        }
        FixedExpansion::new(arr)
    }

    fn set_subexpansion<P2: Property, N2: ArrayLength<f64>>(
        &mut self,
        sub: &FixedExpansion<P2, N2>,
        start: usize
    ) {
        for i in 0..N2::USIZE {
            self.arr[start + i] = sub.arr[i];
        }
    }

    /// Adds another expansion to this expansion
    /// using repeated `grow_expansion`s.
    fn expansion_sum<P2: Property, N2: ArrayLength<f64>>(
        &self,
        other: FixedExpansion<P2, N2>,
    ) -> FixedExpansion<<<P as Min<P2>>::Output as Property>::Weak, <N as Add<N2>>::Output>
        where
            P: Min<P2>,
            N: Add<N2>,
            <N as Add<N2>>::Output: ArrayLength<f64>,
            N: Add<U1>,
            <N as Add<U1>>::Output: ArrayLength<f64>,
    {
        let mut exp = FixedExpansion::<
            <<P as Min<P2>>::Output as Property>::Weak,
            <N as Add<N2>>::Output,
        >::default();
        exp.set_subexpansion(self, 0);
        
        for i in 0..N2::USIZE {
            let sub = exp.subexpansion::<N>(i);
            let sub = sub.grow_expansion(other.arr[i]);
            exp.set_subexpansion(&sub, i);
        }
        
        exp
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

impl<P1: Property, P2: Property, N1: ArrayLength<f64>, N2: ArrayLength<f64>> ExpansionKind<P2, N2>
    for DynamicExpansion<P1, N1>
{
    type Type = DynamicExpansion<P2, N2>;
}

impl<P: Property, N: ArrayLength<f64>> Expansion for DynamicExpansion<P, N> {
    type Prop = P;
    type Bound = N;
    const ZERO_ELIMINATE: bool = true;

    fn arr(&self) -> &GenericArray<f64, Self::Bound> {
        &self.arr
    }

    fn len(&self) -> usize {
        self.len
    }

    fn with_len(arr: impl Into<GenericArray<f64, Self::Bound>>, len: usize) -> Self {
        DynamicExpansion {
            arr: arr.into(),
            len,
            marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use typenum::U3;

    fn e(sig: f64, exp: i32) -> f64 {
        sig * 2f64.powi(exp)
    }

    #[test]
    fn test_fast_two_sum_low_precision() {
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
    fn test_fast_two_sum_high_precision() {
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
    fn test_fast_two_sum_split_b() {
        // b is split in result
        let res = fast_two_sum(1.0, e(11.0, -54));
        assert_eq!(res.arr, [e(-1.0, -54), 1.0 + e(3.0, -52)].into());

        let res = fast_two_sum(1.0, e(15.0, -54));
        assert_eq!(res.arr, [e(-1.0, -54), 1.0 + e(1.0, -50)].into());

        let res = fast_two_sum(1.0, -e(11.0, -54));
        assert_eq!(res.arr, [e(1.0, -54), 1.0 - e(3.0, -52)].into());
    }

    #[test]
    fn test_two_sum_low_precision() {
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
    fn test_two_sum_high_precision() {
        // Result is just [b, a]
        let nums_a = [15.75, 1.5, -4.625, -15.75];
        let nums_b = [
            15.75 * 2f64.powi(-70),
            1.5 * 2f64.powi(-70),
            -4.625 * 2f64.powi(-70),
            -15.75 * 2f64.powi(-70),
        ];
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
    fn test_two_sum_split_b() {
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

    #[test]
    fn test_grow_expansion_zero_fixed() {
        // One or both operands are 0
        let exp_0 = FixedExpansion::<NAdj, U1>::new([0.0]);
        let res = exp_0.grow_expansion(0.0);
        assert_eq!(res.arr, [0.0, 0.0].into());

        let res = exp_0.grow_expansion(5.5);
        assert_eq!(res.arr, [0.0, 5.5].into());

        let exp = FixedExpansion::<NAdj, U2>::new([e(3.0, -20), 3.25]);
        let res = exp.grow_expansion(0.0);
        assert_eq!(res.arr, [0.0, 0.0, e(3.0, -20) + 3.25].into());
    }

    #[test]
    fn test_grow_expansion_zero_dynamic() {
        // One or both operands are 0
        let exp_0 = DynamicExpansion::<NAdj, U1>::with_len([0.0], 0);
        let res = exp_0.grow_expansion(0.0);
        assert_eq!(res.len, 0);

        let res = exp_0.grow_expansion(5.5);
        assert_eq!(res.arr.as_slice()[..1], [5.5]);
        assert_eq!(res.len, 1);

        let exp = DynamicExpansion::<NAdj, U2>::with_len([e(3.0, -20), 3.25], 2);
        let res = exp.grow_expansion(0.0);
        assert_eq!(res.arr.as_slice()[..1], [e(3.0, -20) + 3.25]);
        assert_eq!(res.len, 1);
    }

    #[test]
    fn test_grow_expansion_single_low_precision() {
        // Result can fit in a float
        let exp = FixedExpansion::<NAdj, U1>::new([1.0]);
        let res = exp.grow_expansion(1.75);
        assert_eq!(res.arr, [0.0, 2.75].into());

        let exp = DynamicExpansion::<NAdj, U1>::with_len([1.0], 1);
        let res = exp.grow_expansion(1.75);
        assert_eq!(res.arr.as_slice()[..1], [2.75]);
        assert_eq!(res.len, 1);
    }

    #[test]
    fn test_grow_expansion_single_high_precision() {
        // Result can fit in a float
        let exp = FixedExpansion::<NAdj, U1>::new([1.0]);
        let res = exp.grow_expansion(e(1.0, -53));
        assert_eq!(res.arr, [e(1.0, -53), 1.0].into());

        let exp = DynamicExpansion::<NAdj, U1>::with_len([1.0], 1);
        let res = exp.grow_expansion(e(1.0, -53));
        assert_eq!(res.arr.as_slice()[..2], [e(1.0, -53), 1.0]);
        assert_eq!(res.len, 2);
    }

    #[test]
    fn test_grow_expansion_multiple() {
        // Input is longer
        let exp = FixedExpansion::<NAdj, U2>::new([e(1.0, -60), 1.0]);
        let res = exp.grow_expansion(e(1.0, -30));
        assert_eq!(res.arr, [0.0, e(1.0, -60), e(1.0, -30) + 1.0].into());

        let exp = DynamicExpansion::<NAdj, U2>::with_len([e(1.0, -60), 1.0], 2);
        let res = exp.grow_expansion(e(1.0, -30));
        assert_eq!(res.arr.as_slice()[..2], [e(1.0, -60), e(1.0, -30) + 1.0]);
        assert_eq!(res.len, 2);
    }

    #[test]
    fn test_grow_expansion_cancellation() {
        // Result has zero term in middle
        let exp = FixedExpansion::<NAdj, U3>::new([e(1.0, -106), -e(1.0, -53), 2.0]);
        let res = exp.grow_expansion(e(1.0, -53));
        assert_eq!(res.arr, [e(1.0, -106), 0.0, 0.0, 2.0].into());

        let exp = DynamicExpansion::<NAdj, U3>::with_len([e(1.0, -106), -e(1.0, -53), 2.0], 3);
        let res = exp.grow_expansion(e(1.0, -53));
        assert_eq!(res.arr.as_slice()[..2], [e(1.0, -106), 2.0]);
        assert_eq!(res.len, 2);
    }

    #[test]
    fn test_subexpansion() {
        let exp = FixedExpansion::<NOver, U3>::new([1.0, 2.0, 4.0]);
        let res = exp.subexpansion::<U2>(0);
        assert_eq!(res.arr, [1.0, 2.0].into());
        let res = exp.subexpansion::<U1>(2);
        assert_eq!(res.arr, [4.0].into());
    }

    #[test]
    fn test_set_subexpression() {
        let mut exp = FixedExpansion::<NOver, U3>::new([0.0, 0.0, 0.0]);
        let sub = FixedExpansion::<NAdj, U2>::new([1.0, 2.0]);
        exp.set_subexpansion(&sub, 1);
        assert_eq!(exp.arr, [0.0, 1.0, 2.0].into());
    }
}
