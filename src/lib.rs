//! Implementation of the floating point expansions found in
//! [Adaptive Precision Floating-Point Arithmetic and
//! Fast Robust Geometric Predicates](https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf)
//! in a more Rusty style.
//! Double precision only.
#[macro_use]
extern crate typenum;
#[macro_use]
extern crate generic_array;
extern crate nalgebra;

mod geo;

pub use geo::{orient_2d, orient_3d};

use generic_array::{sequence::Lengthen, ArrayLength, GenericArray};
use std::ops::{Add, Neg, Sub, Div, Index, Mul, RangeTo};
use std::{marker::PhantomData, mem::MaybeUninit};
use typenum::{Greater, Unsigned, U0, U1, U2};

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

/// Takes the weaker of two properties
trait Min<P: Property> {
    type Output: Property;
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

/// Type that can be the length of an expansion.
trait Length: ArrayLength<f64> + ArrayLength<MaybeUninit<f64>> {}
impl<T: ArrayLength<f64> + ArrayLength<MaybeUninit<f64>>> Length for T {}

trait ExpansionKind<P: Property, N: Length> {
    type Type: Expansion;
}

trait PropertyOp<P: Property> {
    type Add;
}
impl<P: Property, P2: Property, N: Length> PropertyOp<P2> for FixedExpansion<P, N>
where
    P: Min<P2>,
{
    type Add = <<P as Min<P2>>::Output as Property>::Weak;
}

impl<P: Property, P2: Property, N: Length> PropertyOp<P2> for DynamicExpansion<P, N>
where
    P: Min<P2>,
    <P as Min<P2>>::Output: StronglyNonoverlapping,
{
    type Add = SNOver;
}

/// The trait representing a generic floating-point expansion.
/// The expansion is assumed to be nonoverlapping and
/// terms are in increasing-magnitude order.
trait Expansion: Default + Index<usize, Output = f64> {
    type Prop: Property;
    type Bound: Length;
    /// Whether to remove zero-terms from expansion
    /// during certain operations
    const ZERO_ELIMINATE: bool;

    /// The number of terms in the expansion.
    fn len(&self) -> usize;

    /// Creates an expansion with a specific length set.
    fn with_len(arr: impl Into<GenericArray<f64, Self::Bound>>, len: usize) -> Self;

    /// Sets the value at a specific index.
    /// Not using `IndexMut` because taking
    /// the reference of an uninitialized value is undefined behavior.
    fn set(&mut self, index: usize, value: f64);

    fn set_len(&mut self, len: usize);

    /// Adds `b` to this expansion,
    /// using a sequence of `two_sum`s.
    fn grow_expansion(&self, b: f64) -> Self::Type
    where
        Self: ExpansionKind<
            <<Self as Expansion>::Prop as Property>::Weak,
            <<Self as Expansion>::Bound as Add<U1>>::Output,
        >,
        <Self as Expansion>::Bound: Add<U1>,
        <<Self as Expansion>::Bound as Add<U1>>::Output: Length,
    {
        let mut exp = Self::Type::default();
        let mut sum = b;
        let mut exp_i = 0;

        for i in 0..self.len() {
            let res = two_sum(sum, self[i]);
            sum = res[1];
            if !Self::ZERO_ELIMINATE || res[0] != 0.0 {
                exp.set(exp_i, res[0]);
                exp_i += 1;
            }
        }

        if !Self::ZERO_ELIMINATE || sum != 0.0 {
            exp.set(exp_i, sum);
            exp_i += 1;
        }

        exp.set_len(exp_i);
        exp
    }

    /// Multiplies this expansion by `b`,
    /// using a sequence of `two_product`s and `two_sum`s.
    /// Note that strongly nonoverlapping is preserved,
    /// as well as the other 2 properties.
    fn scale_expansion(&self, b: f64) -> Self::Type
    where
        Self: ExpansionKind<
            <Self as Expansion>::Prop,
            <<Self as Expansion>::Bound as Mul<U2>>::Output,
        >,
        <Self as Expansion>::Bound: Mul<U2>,
        <<Self as Expansion>::Bound as Mul<U2>>::Output: Length,
    {
        let mut exp = Self::Type::default();
        if self.len() == 0 {
            return exp;
        }

        let mut exp_i = 0;

        let res = two_product(self[0], b);
        let mut prod = res[1];

        if !Self::ZERO_ELIMINATE || res[0] != 0.0 {
            exp.set(exp_i, res[0]);
            exp_i += 1;
        }

        for i in 1..self.len() {
            let res1 = two_product(self[i], b);
            let res2 = two_sum(res1[0], prod);
            let res3 = fast_two_sum(res1[1], res2[1]);
            prod = res3[1];

            for res in vec![res2[0], res3[0]] {
                if !Self::ZERO_ELIMINATE || res != 0.0 {
                    exp.set(exp_i, res);
                    exp_i += 1;
                }
            }
        }

        if !Self::ZERO_ELIMINATE || prod != 0.0 {
            exp.set(exp_i, prod);
            exp_i += 1;
        }

        exp.set_len(exp_i);
        exp
    }

    /// Returns the approximate value of this expansion.
    fn approximate(&self) -> f64 {
        let mut res = 0.0;
        for i in 0..self.len() {
            res += self[i]
        }
        res
    }

    /// Returns the component with the highest magnitude.
    fn highest_magnitude(&self) -> f64 {
        if Self::ZERO_ELIMINATE {
            if self.len() > 0 {
                self[self.len() - 1]
            } else {
                0.0
            }
        } else {
            for i in (0..self.len()).rev() {
                if self[i] != 0.0 {
                    return self[i];
                } 
            }
            0.0
        }
    }
}

/// An expansion whose size is its array size
#[derive(Clone, Debug, Default)]
struct FixedExpansion<P: Property, N: Length> {
    arr: GenericArray<f64, N>,
    marker: PhantomData<P>,
}

impl<P: Property, N: Length> Index<usize> for FixedExpansion<P, N> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < N::USIZE);
        unsafe { &self.arr.get_unchecked(index) }
    }
}

impl<P1: Property, P2: Property, N1: Length, N2: Length> ExpansionKind<P2, N2>
    for FixedExpansion<P1, N1>
{
    type Type = FixedExpansion<P2, N2>;
}

impl<P: Property, N: Length> Expansion for FixedExpansion<P, N> {
    type Prop = P;
    type Bound = N;
    const ZERO_ELIMINATE: bool = false;

    fn len(&self) -> usize {
        N::USIZE
    }

    fn with_len(arr: impl Into<GenericArray<f64, N>>, len: usize) -> Self {
        debug_assert_eq!(len, N::USIZE);
        Self::new(arr)
    }

    fn set(&mut self, index: usize, value: f64) {
        debug_assert!(index < N::USIZE);
        unsafe { *self.arr.get_unchecked_mut(index) = value };
    }

    fn set_len(&mut self, len: usize) {
        debug_assert_eq!(len, N::USIZE)
    }
}

macro_rules! impl_add_fixed {
    ($($a:lifetime)?, $($b:lifetime)?, $($and:tt)?) => {
        impl<$($a,)? $($b,)? P: Property, P2: Property, N: Length, N2: Length> Add<$(&$b)? FixedExpansion<P2, N2>>
            for $(&$a)? FixedExpansion<P, N>
        where
            P: Min<P2>,
            N: Add<N2>,
            <N as Add<N2>>::Output: Length,
            N: Add<U1>,
            <N as Add<U1>>::Output: Length,
        {
            type Output =
                FixedExpansion<<<P as Min<P2>>::Output as Property>::Weak, <N as Add<N2>>::Output>;

            fn add(self, rhs: $(&$b)? FixedExpansion<P2, N2>) -> Self::Output {
                self.expansion_sum($(&$and)? rhs)
            }
        }
    };
}
impl_add_fixed!(  ,   , &);
impl_add_fixed!('a,   , &);
impl_add_fixed!(  , 'b,  );
impl_add_fixed!('a, 'b,  );

macro_rules! impl_neg_fixed {
    ($($a:lifetime)?) => {
        impl<$($a,)? P: Property, N: Length> Neg
            for $(&$a)? FixedExpansion<P, N>
        {
            type Output = FixedExpansion<P, N>;

            fn neg(self) -> Self::Output {
                let mut res = FixedExpansion::<P, N>::default();
                for i in 0..N::USIZE {
                    res.set(i, -self[i]);
                }
                res
            }
        }
    };
}
impl_neg_fixed!(  );
impl_neg_fixed!('a);

macro_rules! impl_sub_fixed {
    ($($a:lifetime)?, $($b:lifetime)?, $($and:tt)?) => {
        impl<$($a,)? $($b,)? P: Property, P2: Property, N: Length, N2: Length> Sub<$(&$b)? FixedExpansion<P2, N2>>
            for $(&$a)? FixedExpansion<P, N>
        where
            P: Min<P2>,
            N: Add<N2>,
            <N as Add<N2>>::Output: Length,
            N: Add<U1>,
            <N as Add<U1>>::Output: Length,
        {
            type Output =
                FixedExpansion<<<P as Min<P2>>::Output as Property>::Weak, <N as Add<N2>>::Output>;

            fn sub(self, rhs: $(&$b)? FixedExpansion<P2, N2>) -> Self::Output {
                self.expansion_sum(&-rhs)
            }
        }
    };
}
impl_sub_fixed!(  ,   , &);
impl_sub_fixed!('a,   , &);
impl_sub_fixed!(  , 'b,  );
impl_sub_fixed!('a, 'b,  );

impl<P: Property, N: Length> FixedExpansion<P, N> {
    fn new(arr: impl Into<GenericArray<f64, N>>) -> Self {
        Self {
            arr: arr.into(),
            marker: PhantomData,
        }
    }

    fn subexpansion<Len: Length>(&self, start: usize) -> FixedExpansion<P, Len> {
        let mut exp = FixedExpansion::<P, Len>::default();
        for i in 0..Len::USIZE {
            exp.set(i, self[start + i]);
        }
        exp
    }

    fn set_subexpansion<P2: Property, N2: Length>(
        &mut self,
        sub: &FixedExpansion<P2, N2>,
        start: usize,
    ) {
        for i in 0..N2::USIZE {
            self.set(start + i, sub[i]);
        }
    }

    /// Adds another expansion to this expansion
    /// using repeated `grow_expansion`s.
    fn expansion_sum<P2: Property, N2: Length>(
        &self,
        other: &FixedExpansion<P2, N2>,
    ) -> FixedExpansion<<<P as Min<P2>>::Output as Property>::Weak, <N as Add<N2>>::Output>
    where
        P: Min<P2>,
        N: Add<N2>,
        <N as Add<N2>>::Output: Length,
        N: Add<U1>,
        <N as Add<U1>>::Output: Length,
    {
        let mut exp = FixedExpansion::<
            <<P as Min<P2>>::Output as Property>::Weak,
            <N as Add<N2>>::Output,
        >::default();
        exp.set_subexpansion(self, 0);

        for i in 0..N2::USIZE {
            let sub = exp.subexpansion::<N>(i);
            let sub = sub.grow_expansion(other[i]);
            exp.set_subexpansion(&sub, i);
        }

        exp
    }

    /// Converts this to a dynamic expression
    fn dynamic(self) -> DynamicExpansion<P, N> {
        DynamicExpansion::with_len(self.arr, N::USIZE)
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

/// Splits the significand in half so prepare it for multiplication.
/// Returns the low bits, then the high bits
fn split(a: f64) -> [f64; 2] {
    let c = (2f64.powi((f64::MANTISSA_DIGITS as i32 + 1) / 2) + 1.0) * a;
    let a_hi = c - (c - a);
    let a_lo = a - a_hi;
    [a_lo, a_hi]
}

/// Constructs an expansion from the product of `a` and `b`.
/// The greater term is `a` * `b` with floating-point multiplication.
fn two_product(a: f64, b: f64) -> FixedExpansion<NAdj, U2> {
    let x = a * b;
    let [a_lo, a_hi] = split(a);
    let [b_lo, b_hi] = split(b);
    let err = x - a_hi * b_hi - a_lo * b_hi - a_hi * b_lo;
    FixedExpansion::new([a_lo * b_lo - err, x])
}

/// An expansion whose size is stored explicitly,
/// but (for now) still has an upper bound for size.
#[derive(Clone, Debug)]
struct DynamicExpansion<P: Property, N: Length> {
    arr: GenericArray<MaybeUninit<f64>, N>,
    len: usize,
    marker: PhantomData<P>,
}

impl<P: Property, N: Length> Default for DynamicExpansion<P, N> {
    fn default() -> Self {
        Self {
            // Safe because we're initialzing `MaybeUninit`s
            arr: unsafe { MaybeUninit::uninit().assume_init() },
            len: 0,
            marker: PhantomData,
        }
    }
}

impl<P: Property, N: Length> Index<usize> for DynamicExpansion<P, N> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < N::USIZE);
        unsafe { &*self.arr.get_unchecked(index).as_ptr() }
    }
}

impl<P1: Property, P2: Property, N1: Length, N2: Length> ExpansionKind<P2, N2>
    for DynamicExpansion<P1, N1>
{
    type Type = DynamicExpansion<P2, N2>;
}

impl<P: Property, N: Length> Expansion for DynamicExpansion<P, N> {
    type Prop = P;
    type Bound = N;
    const ZERO_ELIMINATE: bool = true;

    fn len(&self) -> usize {
        self.len
    }

    fn with_len(arr: impl Into<GenericArray<f64, Self::Bound>>, len: usize) -> Self {
        let arr = arr.into();
        let mut exp = Self::default();
        exp.len = len;
        for i in 0..len {
            exp.set(i, unsafe { *arr.get_unchecked(i) });
        }

        exp
    }

    fn set(&mut self, index: usize, value: f64) {
        debug_assert!(index < N::USIZE);
        unsafe { *self.arr.get_unchecked_mut(index) = MaybeUninit::new(value) };
    }

    fn set_len(&mut self, len: usize) {
        self.len = len;
    }
}

macro_rules! impl_add_dynamic {
    ($($a:lifetime)?, $($b:lifetime)?, $($and:tt)?) => {
        impl<$($a,)? $($b,)? P: Property, P2: Property, N: Length, N2: Length> Add<$(&$b)? DynamicExpansion<P2, N2>>
            for $(&$a)? DynamicExpansion<P, N>
        where
            <P as Min<P2>>::Output: StronglyNonoverlapping,
            P: Min<P2>,
            N: Add<N2>,
            <N as Add<N2>>::Output: Length,
            N: Add<U1>,
            <N as Add<U1>>::Output: Length,
        {
            type Output = DynamicExpansion<SNOver, <N as Add<N2>>::Output>;

            fn add(self, rhs: $(&$b)? DynamicExpansion<P2, N2>) -> Self::Output {
                self.fast_expansion_sum($(&$and)? rhs)
            }
        }
    };
}
impl_add_dynamic!(  ,   , &);
impl_add_dynamic!('a,   , &);
impl_add_dynamic!(  , 'b,  );
impl_add_dynamic!('a, 'b,  );

macro_rules! impl_neg_dynamic {
    ($($a:lifetime)?) => {
        impl<$($a,)? P: Property, N: Length> Neg
            for $(&$a)? DynamicExpansion<P, N>
        {
            type Output = DynamicExpansion<P, N>;

            fn neg(self) -> Self::Output {
                let mut res = DynamicExpansion::<P, N>::default();
                for i in 0..N::USIZE {
                    res.set(i, -self[i]);
                }
                res
            }
        }
    };
}
impl_neg_dynamic!(  );
impl_neg_dynamic!('a);

macro_rules! impl_sub_dynamic {
    ($($a:lifetime)?, $($b:lifetime)?, $($and:tt)?) => {
        impl<$($a,)? $($b,)? P: Property, P2: Property, N: Length, N2: Length> Sub<$(&$b)? DynamicExpansion<P2, N2>>
            for $(&$a)? DynamicExpansion<P, N>
        where
            <P as Min<P2>>::Output: StronglyNonoverlapping,
            P: Min<P2>,
            N: Add<N2>,
            <N as Add<N2>>::Output: Length,
            N: Add<U1>,
            <N as Add<U1>>::Output: Length,
        {
            type Output = DynamicExpansion<SNOver, <N as Add<N2>>::Output>;

            fn sub(self, rhs: $(&$b)? DynamicExpansion<P2, N2>) -> Self::Output {
                self.fast_expansion_sum(&-rhs)
            }
        }
    };
}
impl_sub_dynamic!(  ,   , &);
impl_sub_dynamic!('a,   , &);
impl_sub_dynamic!(  , 'b,  );
impl_sub_dynamic!('a, 'b,  );


impl<P: Property, N: Length> DynamicExpansion<P, N> {
    /// Only to be used for testing
    fn slice(&self) -> &[f64] {
        unsafe { std::mem::transmute::<&[MaybeUninit<f64>], &[f64]>(&self.arr[..self.len]) }
    }

    /// Adds another expansion to this expansion
    /// using a merge and repeated `two_sums`.
    /// Note that the strongly nonoverlapping property is required.
    fn fast_expansion_sum<P2: Property, N2: Length>(
        &self,
        other: &DynamicExpansion<P2, N2>,
    ) -> DynamicExpansion<SNOver, <N as Add<N2>>::Output>
    where
        <P as Min<P2>>::Output: StronglyNonoverlapping,
        P: Min<P2>,
        N: Add<N2>,
        <N as Add<N2>>::Output: Length,
        N: Add<U1>,
        <N as Add<U1>>::Output: Length,
    {
        let mut exp = DynamicExpansion::<SNOver, <N as Add<N2>>::Output>::default();

        // Merge self and other
        let mut i = 0;
        let mut j = 0;
        for k in 0..(self.len + other.len) {
            if i < self.len && (j >= other.len || self[i] < other[j]) {
                exp.set(k, self[i]);
                i += 1;
            } else {
                exp.set(k, other[j]);
                j += 1;
            }
        }

        // Initial fast_two_sum would fail, and besides,
        // the result is correct as is
        if self.len + other.len < 2 {
            exp.len = if self.len + other.len == 1 && exp[0] != 0.0 {
                1
            } else {
                0
            };
            return exp;
        }

        // Do the summing
        let mut exp_i = 0;
        let res = fast_two_sum(exp[1], exp[0]);
        let mut sum = res[1];

        if res[0] != 0.0 {
            exp.set(exp_i, res[0]);
            exp_i += 1;
        }

        for i in 2..(self.len + other.len) {
            let res = two_sum(sum, exp[i]);
            sum = res[1];

            if res[0] != 0.0 {
                exp.set(exp_i, res[0]);
                exp_i += 1;
            }
        }

        if sum != 0.0 {
            exp.set(exp_i, sum);
            exp_i += 1;
        }

        exp.len = exp_i;
        exp
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
        assert_eq!(res.slice(), [] as [f64; 0]);

        let res = exp_0.grow_expansion(5.5);
        assert_eq!(res.slice(), [5.5]);

        let exp = DynamicExpansion::<NAdj, U2>::with_len([e(3.0, -20), 3.25], 2);
        let res = exp.grow_expansion(0.0);
        assert_eq!(res.slice(), [e(3.0, -20) + 3.25]);
    }

    #[test]
    fn test_grow_expansion_single_low_precision() {
        // Result can fit in a float
        let exp = FixedExpansion::<NAdj, U1>::new([1.0]);
        let res = exp.grow_expansion(1.75);
        assert_eq!(res.arr, [0.0, 2.75].into());

        let exp = DynamicExpansion::<NAdj, U1>::with_len([1.0], 1);
        let res = exp.grow_expansion(1.75);
        assert_eq!(res.slice(), [2.75]);
    }

    #[test]
    fn test_grow_expansion_single_high_precision() {
        // Result can't fit in a float
        let exp = FixedExpansion::<NAdj, U1>::new([1.0]);
        let res = exp.grow_expansion(e(1.0, -53));
        assert_eq!(res.arr, [e(1.0, -53), 1.0].into());

        let exp = DynamicExpansion::<NAdj, U1>::with_len([1.0], 1);
        let res = exp.grow_expansion(e(1.0, -53));
        assert_eq!(res.slice(), [e(1.0, -53), 1.0]);
    }

    #[test]
    fn test_grow_expansion_multiple() {
        // Input is longer
        let exp = FixedExpansion::<NAdj, U2>::new([e(1.0, -60), 1.0]);
        let res = exp.grow_expansion(e(1.0, -30));
        assert_eq!(res.arr, [0.0, e(1.0, -60), e(1.0, -30) + 1.0].into());

        let exp = DynamicExpansion::<NAdj, U2>::with_len([e(1.0, -60), 1.0], 2);
        let res = exp.grow_expansion(e(1.0, -30));
        assert_eq!(res.slice(), [e(1.0, -60), e(1.0, -30) + 1.0]);
    }

    #[test]
    fn test_grow_expansion_cancellation() {
        // Result has zero term in middle
        let exp = FixedExpansion::<NAdj, U3>::new([e(1.0, -106), -e(1.0, -53), 2.0]);
        let res = exp.grow_expansion(e(1.0, -53));
        assert_eq!(res.arr, [e(1.0, -106), 0.0, 0.0, 2.0].into());

        let exp = DynamicExpansion::<NAdj, U3>::with_len([e(1.0, -106), -e(1.0, -53), 2.0], 3);
        let res = exp.grow_expansion(e(1.0, -53));
        assert_eq!(res.slice(), [e(1.0, -106), 2.0]);
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
    fn test_set_subexpansion() {
        let mut exp = FixedExpansion::<NOver, U3>::new([0.0, 0.0, 0.0]);
        let sub = FixedExpansion::<NAdj, U2>::new([1.0, 2.0]);
        exp.set_subexpansion(&sub, 1);
        assert_eq!(exp.arr, [0.0, 1.0, 2.0].into());
    }

    #[test]
    fn test_expansion_sum_zero() {
        // One or more operands are zero
        let exp1 = FixedExpansion::<NAdj, U1>::new([0.0]);
        let exp2 = FixedExpansion::<NAdj, U2>::new([0.0, 0.0]);
        let res = exp1.expansion_sum(&exp2);
        assert_eq!(res.arr, [0.0, 0.0, 0.0].into());

        let exp1 = FixedExpansion::<NAdj, U2>::new([1.0, e(1.0, 53)]);
        let res = exp1.expansion_sum(&exp2);
        assert_eq!(res.arr, [0.0, 0.0, 1.0, e(1.0, 53)].into());
        let res = exp2.expansion_sum(&exp1);
        assert_eq!(res.arr, [0.0, 0.0, 1.0, e(1.0, 53)].into());
    }

    #[test]
    fn test_expansion_sum_simple() {
        let exp1 = FixedExpansion::<NAdj, U1>::new([1.0]);
        let exp2 = FixedExpansion::<NAdj, U1>::new([2.0]);
        let res = exp1.expansion_sum(&exp2);
        assert_eq!(res.arr, [0.0, 3.0].into());
        let res = exp2.expansion_sum(&exp1);
        assert_eq!(res.arr, [0.0, 3.0].into());
    }

    #[test]
    fn test_expansion_sum_complex() {
        let exp1 = FixedExpansion::<NAdj, U2>::new([1.0, e(1.0, 53)]);
        let exp2 = FixedExpansion::<NAdj, U2>::new([e(1.0, 53), e(1.0, 106)]);
        let res = exp1.expansion_sum(&exp2);
        assert_eq!(res.arr, [1.0, 0.0, 0.0, e(1.0, 54) + e(1.0, 106)].into());
        let res = exp2.expansion_sum(&exp1);
        assert_eq!(res.arr, [1.0, 0.0, 0.0, e(1.0, 54) + e(1.0, 106)].into());
    }

    #[test]
    fn test_fast_expansion_sum_zero() {
        // One or more operands are zero
        let exp1 = DynamicExpansion::<NAdj, U1>::with_len([0.0], 0);
        let exp2 = DynamicExpansion::<NAdj, U2>::with_len([0.0, 0.0], 1);
        let res = exp1.fast_expansion_sum(&exp1);
        assert_eq!(res.slice(), [] as [f64; 0]); // 0 element merge test case

        let res = exp1.fast_expansion_sum(&exp2);
        assert_eq!(res.slice(), [] as [f64; 0]); // 1 element merge test case

        let exp3 = DynamicExpansion::<NAdj, U1>::with_len([1.0], 1);
        let res = exp1.fast_expansion_sum(&exp3);
        assert_eq!(res.slice(), [1.0]); // 1 element merge test case with nonzero result

        let exp1 = DynamicExpansion::<NAdj, U2>::with_len([1.0, e(1.0, 53)], 2);
        let res = exp1.fast_expansion_sum(&exp2);
        assert_eq!(res.slice(), [1.0, e(1.0, 53)]);
        let res = exp2.fast_expansion_sum(&exp1);
        assert_eq!(res.slice(), [1.0, e(1.0, 53)]);
    }

    #[test]
    fn test_fast_expansion_sum_simple() {
        let exp1 = DynamicExpansion::<NAdj, U1>::with_len([1.0], 1);
        let exp2 = DynamicExpansion::<NAdj, U1>::with_len([2.0], 1);
        let res = exp1.fast_expansion_sum(&exp2);
        assert_eq!(res.slice(), [3.0]);
        let res = exp2.fast_expansion_sum(&exp1);
        assert_eq!(res.slice(), [3.0]);
    }

    #[test]
    fn test_fast_expansion_sum_complex() {
        let exp1 = DynamicExpansion::<NAdj, U2>::with_len([1.0, e(1.0, 53)], 2);
        let exp2 = DynamicExpansion::<NAdj, U2>::with_len([e(1.0, 53), e(1.0, 106)], 2);
        let res = exp1.fast_expansion_sum(&exp2);
        assert_eq!(res.slice(), [1.0, e(1.0, 54) + e(1.0, 106)]);
        let res = exp2.fast_expansion_sum(&exp1);
        assert_eq!(res.slice(), [1.0, e(1.0, 54) + e(1.0, 106)]);
    }

    #[test]
    fn test_fast_expansion_sum_cancellation() {
        // Result has zero term in middle
        let exp1 = DynamicExpansion::<NAdj, U3>::with_len([e(1.0, -106), -e(1.0, -53), 2.0], 3);
        let exp2 = DynamicExpansion::<NAdj, U1>::with_len([e(1.0, -53)], 1);
        let res = exp1.fast_expansion_sum(&exp2);
        assert_eq!(res.slice(), [e(1.0, -106), 2.0]);
        let res = exp2.fast_expansion_sum(&exp1);
        assert_eq!(res.slice(), [e(1.0, -106), 2.0]);
    }

    #[test]
    fn test_two_product_low_precision() {
        // Result can fit in f64
        let nums = [15.75, 1.5, 0.0, -4.625, -15.75];
        for a in &nums {
            for b in &nums {
                let res = two_product(*a, *b);
                assert_eq!(res.arr, [0.0, a * b].into());
            }
        }
    }

    #[test]
    fn test_two_product_high_precision() {
        // Result takes 2 f64s
        let res = two_product(1.0 + e(1.0, 27), 1.0 + e(1.0, 30));
        assert_eq!(res.arr, [1.0, e(1.0, 27) + e(1.0, 30) + e(1.0, 57)].into());
    }

    #[test]
    fn test_scale_expansion_zero_fixed() {
        // One or both operands are 0
        let exp_0 = FixedExpansion::<NAdj, U1>::new([0.0]);
        let res = exp_0.scale_expansion(0.0);
        assert_eq!(res.arr, [0.0, 0.0].into());

        let res = exp_0.scale_expansion(5.5);
        assert_eq!(res.arr, [0.0, 0.0].into());

        let exp = FixedExpansion::<NAdj, U2>::new([e(3.0, -20), 3.25]);
        let res = exp.scale_expansion(0.0);
        assert_eq!(res.arr, [0.0, 0.0, 0.0, 0.0].into());
    }

    #[test]
    fn test_scale_expansion_zero_dynamic() {
        // One or both operands are 0
        let exp_0 = DynamicExpansion::<NAdj, U1>::with_len([0.0], 0);
        let res = exp_0.scale_expansion(0.0);
        assert_eq!(res.slice(), [] as [f64; 0]);

        let res = exp_0.scale_expansion(5.5);
        assert_eq!(res.slice(), [] as [f64; 0]);

        let exp = DynamicExpansion::<NAdj, U2>::with_len([e(3.0, -20), 3.25], 2);
        let res = exp.scale_expansion(0.0);
        assert_eq!(res.slice(), [] as [f64; 0]);
    }

    #[test]
    fn test_scale_expansion_single_low_precision() {
        // Result can fit in a float
        let exp = FixedExpansion::<NAdj, U1>::new([2.0]);
        let res = exp.scale_expansion(1.75);
        assert_eq!(res.arr, [0.0, 3.5].into());

        let exp = DynamicExpansion::<NAdj, U1>::with_len([2.0], 1);
        let res = exp.scale_expansion(1.75);
        assert_eq!(res.slice(), [3.5]);
    }

    #[test]
    fn test_scale_expansion_single_high_precision() {
        // Result can't fit in a float
        let exp = FixedExpansion::<NAdj, U1>::new([1.5]);
        let res = exp.scale_expansion(e(1.0, -52) + 1.0);
        assert_eq!(res.arr, [e(-1.0, -53), 1.5 + e(1.0, -51)].into());

        let exp = DynamicExpansion::<NAdj, U1>::with_len([1.5], 1);
        let res = exp.scale_expansion(e(1.0, -52) + 1.0);
        assert_eq!(res.slice(), [e(-1.0, -53), 1.5 + e(1.0, -51)]);
    }

    #[test]
    fn test_scale_expansion_multiple() {
        // Input is longer
        let exp = FixedExpansion::<NAdj, U2>::new([e(1.0, -53), 1.0]);
        let res = exp.scale_expansion(5.0);
        assert_eq!(res.arr, [0.0, 0.0, e(-3.0, -53), 5.0 + e(1.0, -50)].into());

        let exp = DynamicExpansion::<NAdj, U2>::with_len([e(1.0, -53), 1.0], 2);
        let res = exp.scale_expansion(5.0);
        assert_eq!(res.slice(), [e(-3.0, -53), 5.0 + e(1.0, -50)]);
    }

    #[test]
    fn test_highest_magnitude_zero() {
        let exp = FixedExpansion::<NAdj, U2>::new([0.0, 0.0]);
        let res = exp.highest_magnitude();
        assert_eq!(res, 0.0);

        let exp = DynamicExpansion::<NAdj, U1>::with_len([0.0], 0);
        let res = exp.highest_magnitude();
        assert_eq!(res, 0.0);
    }

    #[test]
    fn test_highest_magnitude_positive() {
        let exp = FixedExpansion::<NAdj, U2>::new([1.0, e(1.0, 53)]);
        let res = exp.highest_magnitude();
        assert_eq!(res, e(1.0, 53));

        let exp = DynamicExpansion::<NAdj, U2>::with_len([1.0, e(1.0, 53)], 2);
        let res = exp.highest_magnitude();
        assert_eq!(res, e(1.0, 53));
    }

    #[test]
    fn test_highest_magnitude_negative() {
        let exp = FixedExpansion::<NAdj, U2>::new([1.0, e(-1.0, 53)]);
        let res = exp.highest_magnitude();
        assert_eq!(res, e(-1.0, 53));

        let exp = DynamicExpansion::<NAdj, U2>::with_len([1.0, e(-1.0, 53)], 2);
        let res = exp.highest_magnitude();
        assert_eq!(res, e(-1.0, 53));
    }
    
    #[test]
    fn test_highest_magnitude_top_zero() {
        let exp = FixedExpansion::<NAdj, U3>::new([1.0, e(-1.0, 53), 0.0]);
        let res = exp.highest_magnitude();
        assert_eq!(res, e(-1.0, 53));
    }
}
