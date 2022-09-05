use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::iter::FromIterator;

use num_traits::float::Float;
use rand::Rng;

use crate::distribution::{Continuous, ContinuousCDF, Uniform};
use crate::statistics::{Distribution, Max, Min};
use crate::{Result, StatsError};

#[derive(Clone, Copy, Debug, PartialEq)]
struct NonNAN<T>(T);

impl<T: PartialEq> Eq for NonNAN<T> {}

impl<T: PartialOrd> PartialOrd for NonNAN<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T: PartialOrd> Ord for NonNAN<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Implements the [Empirical
/// Distribution](https://en.wikipedia.org/wiki/Empirical_distribution_function)
///
/// # Examples
///
/// ```
/// use statrs::distribution::{Continuous, Empirical};
/// use statrs::statistics::Distribution;
///
/// let samples = vec![0.0, 5.0, 10.0];
///
/// let empirical = Empirical::from_vec(samples);
/// assert_eq!(empirical.mean().unwrap(), 5.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalWeighted {
    sumw: f64,
    /// See West's paper:  https://dl.acm.org/doi/pdf/10.1145/359146.359153
    m_t: Option<(f64, f64)>,
    // keys are data points, values are number of data points with equal value
    data: BTreeMap<NonNAN<f64>, i64>,
}

impl EmpiricalWeighted {
    /// Constructs a new empty empirical distribution.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::Empirical;
    ///
    /// let mut result = Empirical::new();
    /// assert!(result.is_ok());
    ///
    /// ```
    pub fn new() -> Result<Self> {
        Ok(Self {
            sumw: 0.,
            m_t: None,
            data: BTreeMap::new(),
        })
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.sumw as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[deprecated = "Use the newer From<AsRef<[f64]>> instead."]
    pub fn from_vec(src: Vec<f64>) -> Self {
        let mut empirical = Self::new().unwrap();
        for elt in src.into_iter() {
            empirical.add(elt);
        }
        empirical
    }

    #[inline(always)]
    pub fn add(&mut self, data_point: f64) {
        self.add_with_weight_west(data_point, 1)
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // See:
    // - https://dl.acm.org/doi/pdf/10.1145/359146.359153 (West's paper)
    // - Note that West's `SUMW` is actually our `self.sumw`.
    fn update_with_weight_west(&mut self, data_point: f64, weight: i64) {
        let w = weight as f64;
        self.m_t = Some(if let Some((m, t)) = self.m_t {
            let q = data_point - m;
            // SAFETY: Division by zero cannot occur because (for now, that negative weights are
            // not really supported by the API) `weight` is always passed by `add*` and `remove*`
            // methods. `add*` cannot go wrong, and `remove*` ensure its correctness.
            let r = q * w / (self.sumw + w);
            let new_m = m + r;
            let new_t = t + r * self.sumw * q;
            (new_m, new_t)
        } else {
            (data_point, 0.)
        });
        self.sumw += w;

        *self.data.entry(NonNAN(data_point)).or_insert(0) += weight;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////
    pub fn add_with_weight_west(&mut self, data_point: f64, weight: u64) {
        if data_point.is_nan() || weight == 0 {
            return;
        }
        self.update_with_weight_west(
            data_point,
            i64::try_from(weight).expect("failed to convert u64 `weight` to i64"),
        )
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////
    pub fn remove_with_weight_west(&mut self, data_point: f64, weight: u64) {
        if data_point.is_nan() {
            return;
        }

        let key = NonNAN(data_point);
        if let Some(&old_weight) = self.data.get(&key) {
            if old_weight as u64 == weight && self.data.len() == 1 {
                debug_assert_eq!(
                    weight as f64, self.sumw,
                    "please file a BUG REPORT for this"
                );
                let _ = self.data.remove(&key);
                self.m_t = None;
                self.sumw = 0.;
                return;
            }
        }
        self.update_with_weight_west(
            data_point,
            -i64::try_from(weight).expect("failed to convert u64 `weight` to negative i64"),
        )
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////
    #[inline(always)]
    pub fn remove_west(&mut self, data_point: f64) {
        self.remove_with_weight_west(data_point, 1)

        //if data_point.is_nan() {
        //    return;
        //}
        //
        //let key = NonNAN(data_point);
        //if let Some(&old_weight) = self.data.get(&key) {
        //    if old_weight == 1 && self.data.len() == 1 {
        //        debug_assert_eq!(1., self.sumw, "please file a BUG REPORT for this");
        //        let _ = self.data.remove(&key);
        //        self.m_t = None;
        //        self.sumw = 0.;
        //        return;
        //    }
        //}
        //self.update_with_weight_west(data_point, -1)

        //if let Some(old_weight) = self.data.remove(&NonNAN(data_point)) {
        //    if old_weight == 1 {
        //        if self.data.is_empty() {
        //            debug_assert_eq!(1., self.sumw, "please file a BUG REPORT for this");
        //            self.m_t = None;
        //            self.sumw = 0.;
        //            return;
        //        }
        //    } else {
        //        self.data.insert(NonNAN(data_point), old_weight);
        //    }
        //
        //    self.update_with_weight_west(data_point, -1)
        //}
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////

    //pub fn add_with_weight(&mut self, data_point: f64, weight: u64) {
    //    // We should probably avoid the calculations below for unweighted input
    //    if weight == 1 {
    //        return self.add(data_point);
    //    }
    //
    //    if data_point.is_nan() {
    //        return;
    //    }
    //
    //    self.mean_and_var = self
    //        .mean_and_var
    //        .map_or(Some((data_point, 0.)), |(mean, var)| {
    //            // Auxiliary renaming for convenience:
    //            let v = data_point;
    //            let n = self.sumw;
    //            let k = weight as f64;
    //            // Repeated calculations:
    //            let psi_n_1 = digamma(n + 1.);
    //            let psi_n_k = digamma(n + k);
    //
    //            // New variance consists of addends s1, s2 and s3
    //            let s1 = v * v * n / (n + k);
    //            let s2a = n * n * mean * mean * k / (n * (n + k));
    //            let s2b = v
    //                * v
    //                * (((k - 1.) * (k + 2. * n + 1.)) / (n + k) - ((2. * n + 1.) * psi_n_k)
    //                    + ((2. * n + 1.) * psi_n_1));
    //            let s2c = ((1. - k) / (n + k)) + psi_n_k - psi_n_1;
    //            let s2 = s2a + s2b + s2c;
    //            let s3 = 2.
    //                * v
    //                * (-digamma(n + k + 1.) * ((n + 1.) * v * v - n * mean)
    //                    + k * v * v
    //                    + psi_n_1 * ((n + 1.) * v * v - n * mean));
    //            let new_var = var + s1 + s2 - s3;
    //
    //            let new_mean = (n * mean + k * v) / (n + k);
    //
    //            Some((new_mean, new_var))
    //        });
    //    self.sumw += weight as f64;
    //
    //    *self.data.entry(NonNAN(data_point)).or_insert(0) += weight as i64;
    //}

    // Due to issues with rounding and floating-point accuracy the default
    // implementation may be ill-behaved.
    // Specialized inverse cdfs should be used whenever possible.
    // Performs a binary search on the domain of `cdf` to obtain an approximation
    // of `F^-1(p) := inf { x | F(x) >= p }`. Needless to say, performance may
    // may be lacking.
    // This function is identical to the default method implementation in the
    // `ContinuousCDF` trait and is used to implement the rand trait `Distribution`.
    fn __inverse_cdf(&self, p: f64) -> f64 {
        if p == 0.0 {
            return self.min();
        };
        if p == 1.0 {
            return self.max();
        };
        let mut high = 2.0;
        let mut low = -high;
        while self.cdf(low) > p {
            low = low + low;
        }
        while self.cdf(high) < p {
            high = high + high;
        }
        let mut i = 16;
        while i != 0 {
            let mid = (high + low) / 2.0;
            if self.cdf(mid) >= p {
                high = mid;
            } else {
                low = mid;
            }
            i -= 1;
        }
        (high + low) / 2.0
    }
}

//impl<V> From<V> for EmpiricalWeighted
//where
//    V: AsRef<[f64]>,
//{
//    fn from(v: V) -> Self {
//        let mut empirical = Self::new().expect("failed creating new EmpiricalWeighted");
//        for elt in v.as_ref() {
//            empirical.add(*elt);
//        }
//        empirical
//    }
//}

impl FromIterator<f64> for EmpiricalWeighted {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let mut ret = Self::new().expect("failed to initialize EmpiricalWeighted");
        for v in iter {
            //ret.add(v);
            ret.add_with_weight_west(v, 1)
        }
        ret
    }
}

impl FromIterator<(f64, u64)> for EmpiricalWeighted {
    fn from_iter<T: IntoIterator<Item = (f64, u64)>>(iter: T) -> Self {
        let mut ret = Self::new().expect("failed to initialize EmpiricalWeighted");
        for (data_point, weight) in iter {
            //for _ in 0..weight {
            //    ret.add(data_point);
            //}
            ret.add_with_weight_west(data_point, weight);
        }
        ret
    }
}

//impl<V, W> From<(V, Option<W>)> for EmpiricalWeighted
//where
//    V: AsRef<[f64]>,
//    W: AsRef<[usize]>,
//{
//    fn from((data_points, weights): (V, Option<W>)) -> Self {
//        let dp = data_points.as_ref();
//        //let _w = weights.map_or_else(
//        //    || Cow::Owned(vec![1, dp.len()]),
//        //    |w| {
//        //        let weights = w.as_ref();
//        //        Cow::Borrowed(weights)
//        //    },
//        //);
//        let mut w_aux = Vec::new();
//        let w = if let Some(w) = weights {
//            //let weights = w.as_ref();
//            //Cow::Borrowed(&weights)
//            //
//            Cow::Borrowed(&w.as_ref())
//        } else {
//            //let weights = vec![1usize, dp.len()];
//            //Cow::Owned(weights.as_ref())
//            //
//            //Cow::Owned(vec![1usize, dp.len()].as_ref())
//            w_aux = vec![1usize, dp.len()];
//            Cow::Owned(w_aux.as_ref())
//        };
//        //if let Some(w) = weights {
//        //    assert_eq!(
//        //        dp.len(),
//        //        w.as_ref().len(),
//        //        "data_points and weights must be of the same length"
//        //    );
//        //}
//
//        todo!()
//    }
//}

//impl<V> TryFrom<V> for EmpiricalWeighted
//where
//    V: AsRef<[f64]>,
//{
//    type Error = crate::StatsError;
//
//    fn try_from(v: V) -> std::result::Result<Self, Self::Error> {
//        let mut empirical = Self::new()?;
//        for elt in v.as_ref() {
//            empirical.add(*elt);
//        }
//        Ok(empirical)
//    }
//}

impl ::rand::distributions::Distribution<f64> for EmpiricalWeighted {
    fn sample<R: ?Sized + Rng>(&self, rng: &mut R) -> f64 {
        let uniform = Uniform::new(0.0, 1.0).unwrap();
        self.__inverse_cdf(uniform.sample(rng))
    }
}

/// # Panics
///
/// - When sample's population is zero.
impl Max<f64> for EmpiricalWeighted {
    fn max(&self) -> f64 {
        //self.data.iter().rev().map(|(key, _)| key.0).next().unwrap()
        //(*self.data.iter().last().unwrap().0).0
        self.data.iter().map(|(key, _)| key.0).last().unwrap()
    }
}

/// # Panics
///
/// - When sample's population is zero.
impl Min<f64> for EmpiricalWeighted {
    fn min(&self) -> f64 {
        //self.data.iter().map(|(key, _)| key.0).next().unwrap()
        (*self.data.iter().next().unwrap().0).0
    }
}

impl Distribution<f64> for EmpiricalWeighted {
    fn mean(&self) -> Option<f64> {
        self.m_t.map(|(m, _t)| m)
    }

    fn variance(&self) -> Option<f64> {
        // Prevent division by zero: if `self.sumw` == 1, the variance is zero.
        if self.sumw as usize == 1 {
            Some(0.)
        } else {
            self.m_t
                .map(|(_m, t)| t * self.sumw / ((self.sumw - 1.) * self.sumw))
        }
    }
}

impl ContinuousCDF<f64, f64> for EmpiricalWeighted {
    fn cdf(&self, x: f64) -> f64 {
        let mut sum = 0;
        for (key, weight) in &self.data {
            if key.0 > x {
                return sum as f64 / self.sumw;
            }
            sum += weight;
        }
        sum as f64 / self.sumw
    }

    fn sf(&self, x: f64) -> f64 {
        let mut sum = 0;
        for (key, weight) in self.data.iter().rev() {
            if key.0 <= x {
                return sum as f64 / self.sumw;
            }
            sum += weight;
        }
        sum as f64 / self.sumw
    }
}

//#[cfg(all(test, feature = "nightly"))]
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn test_cdf_from_vec() {
        let samples = vec![5.0, 10.0];
        #[allow(deprecated)]
        let mut empirical = EmpiricalWeighted::from_vec(samples);
        test_cdf(&mut empirical);
    }

    #[test]
    fn test_cdf_from_iter() {
        //let mut empirical: EmpiricalWeighted = vec![5., 10.].into();
        //let mut empirical: EmpiricalWeighted = (vec![5., 10.], Option::<&[usize]>::None).into();
        let mut empirical = EmpiricalWeighted::from_iter(vec![5., 10.].into_iter());
        test_cdf(&mut empirical);
    }

    #[test]
    fn test_sf_from_vec() {
        let samples = vec![5.0, 10.0];
        #[allow(deprecated)]
        let mut empirical = EmpiricalWeighted::from_vec(samples);
        test_sf(&mut empirical);
    }

    #[test]
    fn test_sf_from_iter() {
        //let mut empirical: EmpiricalWeighted = vec![5., 10.].into();
        //let mut empirical: EmpiricalWeighted = (vec![5., 10.], Option::<&[usize]>::None).into();
        let mut empirical = EmpiricalWeighted::from_iter(vec![5., 10.].into_iter());
        test_sf(&mut empirical);
    }

    //#[test]
    //fn test_cdf_from_iter_weights() {
    //    let weighted_values = HashMap::new();
    //    weighted_values.insert(5., 2);
    //    weighted_values.insert(10., 2);

    //    let mut empirical = EmpiricalWeighted::from_iter(weighted_values);
    //    test_cdf(&mut empirical);
    //}

    fn test_cdf(empirical: &mut EmpiricalWeighted) {
        assert_eq!(empirical.cdf(0.0), 0.0);
        assert_eq!(empirical.cdf(5.0), 0.5);
        assert_eq!(empirical.cdf(5.5), 0.5);
        assert_eq!(empirical.cdf(6.0), 0.5);
        assert_eq!(empirical.cdf(10.0), 1.0);
        assert_eq!(empirical.min(), 5.0);
        assert_eq!(empirical.max(), 10.0);
        //empirical.add(2.0);
        //empirical.add(2.0);
        //empirical.add_with_weight(2.0, 2);
        empirical.add_with_weight_west(2.0, 2);
        assert_eq!(empirical.cdf(0.0), 0.0);
        assert_eq!(empirical.cdf(5.0), 0.75);
        assert_eq!(empirical.cdf(5.5), 0.75);
        assert_eq!(empirical.cdf(6.0), 0.75);
        assert_eq!(empirical.cdf(10.0), 1.0);
        assert_eq!(empirical.min(), 2.0);
        assert_eq!(empirical.max(), 10.0);
        let unchanged = empirical.clone();
        //empirical.add(2.0);
        //empirical.add_with_weight(2.0, 1);
        //empirical.remove(2.0);
        empirical.add_with_weight_west(2.0, 1);
        empirical.remove_west(2.0);
        // because of rounding errors, this doesn't hold in general
        // due to the mean and variance being calculated in a streaming way
        assert_eq!(unchanged, *empirical);

        eprintln!(
            "\n{empirical:?}, mean = {:?}, var = {:?}\n",
            empirical.mean(),
            empirical.variance()
        );
    }

    fn test_sf(empirical: &mut EmpiricalWeighted) {
        assert_eq!(empirical.sf(0.0), 1.0);
        assert_eq!(empirical.sf(5.0), 0.5);
        assert_eq!(empirical.sf(5.5), 0.5);
        assert_eq!(empirical.sf(6.0), 0.5);
        assert_eq!(empirical.sf(10.0), 0.0);
        assert_eq!(empirical.min(), 5.0);
        assert_eq!(empirical.max(), 10.0);
        //empirical.add(2.0);
        //empirical.add(2.0);
        empirical.add_with_weight_west(2.0, 2);
        assert_eq!(empirical.sf(0.0), 1.0);
        assert_eq!(empirical.sf(5.0), 0.25);
        assert_eq!(empirical.sf(5.5), 0.25);
        assert_eq!(empirical.sf(6.0), 0.25);
        assert_eq!(empirical.sf(10.0), 0.0);
        assert_eq!(empirical.min(), 2.0);
        assert_eq!(empirical.max(), 10.0);
        let unchanged = empirical.clone();
        //empirical.add(2.0);
        //empirical.remove(2.0);
        empirical.add_with_weight_west(2.0, 1);
        empirical.remove_west(2.0);
        // because of rounding errors, this doesn't hold in general
        // due to the mean and variance being calculated in a streaming way
        assert_eq!(unchanged, *empirical);

        eprintln!(
            "\n{empirical:?}, mean = {:?}, var = {:?}\n",
            empirical.mean(),
            empirical.variance()
        );
    }
}
