use std::{
    marker::PhantomData,
    ops::{Add, Mul, Neg, Sub},
};

use group::ff::Field;
use pasta_curves::arithmetic::FieldExt;

use super::{
    Basis, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
};
use crate::arithmetic::parallelize;

/// A reference to a polynomial registered with an [`Evaluator`].
#[derive(Clone)]
pub(crate) struct AstLeaf<E, B: Basis> {
    index: usize,
    rotation: Rotation,
    _evaluator: PhantomData<(E, B)>,
}

impl<E, B: Basis> AstLeaf<E, B> {
    /// Produces a new `AstLeaf` node corresponding to the underlying polynomial at a
    /// _new_ rotation. Existing rotations applied to this leaf node are ignored and the
    /// returned polynomial is not rotated _relative_ to the previous structure.
    pub(crate) fn with_rotation(&self, rotation: Rotation) -> Self {
        AstLeaf {
            index: self.index,
            rotation,
            _evaluator: PhantomData::default(),
        }
    }
}

/// An evaluation context for polynomial operations.
///
/// This context enables us to de-duplicate queries of circuit columns (and the rotations
/// they might require), by storing a list of all the underlying polynomials involved in
/// any query (which are almost certainly column polynomials). We use the context like so:
///
/// - We register each underlying polynomial with the evaluator, which returns a reference
///   to it as a [`AstLeaf`].
/// - The references are then used to build up a [`Ast`] that represents the overall
///   operations to be applied to the polynomials.
/// - Finally, we call [`Evaluator::evaluate`] passing in the [`Ast`].
pub(crate) struct Evaluator<E, F: Field, B: Basis> {
    polys: Vec<Polynomial<F, B>>,
    _context: E,
}

/// Constructs a new `Evaluator`.
///
/// The `context` parameter is used to provide type safety for evaluators. It ensures that
/// an evaluator will only be used to evaluate [`Ast`]s containing [`AstLeaf`]s obtained
/// from itself. It should be set to the empty closure `|| {}`, because anonymous closures
/// all have unique types.
pub(crate) fn new_evaluator<E: Fn() + Clone, F: Field, B: Basis>(context: E) -> Evaluator<E, F, B> {
    Evaluator {
        polys: vec![],
        _context: context,
    }
}

impl<E, F: Field, B: Basis> Evaluator<E, F, B> {
    /// Registers the given polynomial for use in this evaluation context.
    pub(crate) fn register_poly(&mut self, poly: Polynomial<F, B>) -> AstLeaf<E, B> {
        let index = self.polys.len();
        self.polys.push(poly);

        AstLeaf {
            index,
            rotation: Rotation::cur(),
            _evaluator: PhantomData::default(),
        }
    }

    /// Evaluates the given polynomial operation against this context.
    pub(crate) fn evaluate(
        &self,
        ast: &Ast<E, F, B>,
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, B>
    where
        F: FieldExt,
        B: BasisOps,
    {
        match ast {
            Ast::Poly(AstLeaf {
                index, rotation, ..
            }) => B::rotate(domain, &self.polys[*index], *rotation),
            Ast::Add(a, b) => {
                let a = self.evaluate(a, domain);
                let b = self.evaluate(b, domain);
                a + &b
            }
            Ast::Mul(AstMul(a, b)) => {
                let a = self.evaluate(a, domain);
                let b = self.evaluate(b, domain);
                B::mul(domain, a, b)
            }
            Ast::Scale(a, scalar) => {
                let a = self.evaluate(a, domain);
                a * *scalar
            }
            Ast::LinearTerm(scalar) => B::linear_term(domain, *scalar),
            Ast::ConstantTerm(scalar) => B::constant_term(domain, *scalar),
        }
    }
}

/// Struct representing the [`Ast::Mul`] case.
///
/// This struct exists to make the internals of this case private so that we don't
/// accidentally construct this case directly, because it can only be implemented for the
/// [`ExtendedLagrangeCoeff`] basis.
#[derive(Clone)]
pub(crate) struct AstMul<E, F: Field, B: Basis>(Box<Ast<E, F, B>>, Box<Ast<E, F, B>>);

/// A polynomial operation backed by an [`Evaluator`].
#[derive(Clone)]
pub(crate) enum Ast<E, F: Field, B: Basis> {
    Poly(AstLeaf<E, B>),
    Add(Box<Ast<E, F, B>>, Box<Ast<E, F, B>>),
    Mul(AstMul<E, F, B>),
    Scale(Box<Ast<E, F, B>>, F),
    /// The degree-1 term of a polynomial.
    ///
    /// The field element is the coeffient of the term in the standard basis, not the
    /// coefficient basis.
    LinearTerm(F),
    /// The degree-0 term of a polynomial.
    ///
    /// The field element is the same in both the standard and evaluation bases.
    ConstantTerm(F),
}

impl<E, F: Field, B: Basis> From<AstLeaf<E, B>> for Ast<E, F, B> {
    fn from(leaf: AstLeaf<E, B>) -> Self {
        Ast::Poly(leaf)
    }
}

impl<E, F: Field, B: Basis> Neg for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn neg(self) -> Self::Output {
        Ast::Scale(Box::new(self), -F::one())
    }
}

impl<E: Clone, F: Field, B: Basis> Neg for &Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn neg(self) -> Self::Output {
        -(self.clone())
    }
}

impl<E, F: Field, B: Basis> Add<Ast<E, F, B>> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn add(self, other: Ast<E, F, B>) -> Self::Output {
        Ast::Add(Box::new(self), Box::new(other))
    }
}

impl<'a, E: Clone, F: Field, B: Basis> Add<&'a Ast<E, F, B>> for &'a Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn add(self, other: &'a Ast<E, F, B>) -> Self::Output {
        self.clone() + other.clone()
    }
}

impl<E, F: Field, B: Basis> Sub<Ast<E, F, B>> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn sub(self, other: Ast<E, F, B>) -> Self::Output {
        self + (-other)
    }
}

impl<'a, E: Clone, F: Field, B: Basis> Sub<&'a Ast<E, F, B>> for &'a Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn sub(self, other: &'a Ast<E, F, B>) -> Self::Output {
        self + &(-other)
    }
}

impl<E, F: Field> Mul<Ast<E, F, LagrangeCoeff>> for Ast<E, F, LagrangeCoeff> {
    type Output = Ast<E, F, LagrangeCoeff>;

    fn mul(self, other: Ast<E, F, LagrangeCoeff>) -> Self::Output {
        Ast::Mul(AstMul(Box::new(self), Box::new(other)))
    }
}

impl<'a, E: Clone, F: Field> Mul<&'a Ast<E, F, LagrangeCoeff>> for &'a Ast<E, F, LagrangeCoeff> {
    type Output = Ast<E, F, LagrangeCoeff>;

    fn mul(self, other: &'a Ast<E, F, LagrangeCoeff>) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<E, F: Field> Mul<Ast<E, F, ExtendedLagrangeCoeff>> for Ast<E, F, ExtendedLagrangeCoeff> {
    type Output = Ast<E, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: Ast<E, F, ExtendedLagrangeCoeff>) -> Self::Output {
        Ast::Mul(AstMul(Box::new(self), Box::new(other)))
    }
}

impl<'a, E: Clone, F: Field> Mul<&'a Ast<E, F, ExtendedLagrangeCoeff>>
    for &'a Ast<E, F, ExtendedLagrangeCoeff>
{
    type Output = Ast<E, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: &'a Ast<E, F, ExtendedLagrangeCoeff>) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<E, F: Field, B: Basis> Mul<F> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn mul(self, other: F) -> Self::Output {
        Ast::Scale(Box::new(self), other)
    }
}

impl<E: Clone, F: Field, B: Basis> Mul<F> for &Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn mul(self, other: F) -> Self::Output {
        Ast::Scale(Box::new(self.clone()), other)
    }
}

/// Operations which can be performed over a given basis.
pub(crate) trait BasisOps: Basis {
    fn constant_term<F: FieldExt>(domain: &EvaluationDomain<F>, scalar: F) -> Polynomial<F, Self>;
    fn linear_term<F: FieldExt>(domain: &EvaluationDomain<F>, scalar: F) -> Polynomial<F, Self>;
    fn rotate<F: FieldExt>(
        domain: &EvaluationDomain<F>,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Polynomial<F, Self>;
    fn mul<F: FieldExt>(
        domain: &EvaluationDomain<F>,
        a: Polynomial<F, Self>,
        b: Polynomial<F, Self>,
    ) -> Polynomial<F, Self>;
}

impl BasisOps for Coeff {
    fn constant_term<F: FieldExt>(domain: &EvaluationDomain<F>, scalar: F) -> Polynomial<F, Self> {
        let mut poly = domain.empty_coeff();
        poly[0] = scalar;
        poly
    }

    fn linear_term<F: FieldExt>(domain: &EvaluationDomain<F>, scalar: F) -> Polynomial<F, Self> {
        todo!()
    }

    fn rotate<F: FieldExt>(
        _: &EvaluationDomain<F>,
        _: &Polynomial<F, Self>,
        _: Rotation,
    ) -> Polynomial<F, Self> {
        panic!("Can't rotate polynomials in the standard basis")
    }

    fn mul<F: FieldExt>(
        _: &EvaluationDomain<F>,
        _: Polynomial<F, Self>,
        _: Polynomial<F, Self>,
    ) -> Polynomial<F, Self> {
        panic!("Can't multiply polynomials in the standard basis")
    }
}

impl BasisOps for LagrangeCoeff {
    fn constant_term<F: FieldExt>(domain: &EvaluationDomain<F>, scalar: F) -> Polynomial<F, Self> {
        domain.constant_lagrange(scalar)
    }

    fn linear_term<F: FieldExt>(domain: &EvaluationDomain<F>, scalar: F) -> Polynomial<F, Self> {
        todo!()
    }

    fn rotate<F: FieldExt>(
        _: &EvaluationDomain<F>,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Polynomial<F, Self> {
        poly.rotate(rotation)
    }

    fn mul<F: FieldExt>(
        domain: &EvaluationDomain<F>,
        a: Polynomial<F, Self>,
        b: Polynomial<F, Self>,
    ) -> Polynomial<F, Self> {
        let mut modified_a: Vec<_> = domain
            .empty_lagrange()
            .values
            .into_iter()
            .map(|_| F::one())
            .collect();
        parallelize(&mut modified_a, |modified_a, start| {
            for ((modified_a, a), b) in modified_a
                .iter_mut()
                .zip(a[start..].iter())
                .zip(b[start..].iter())
            {
                *modified_a *= *a * b;
            }
        });
        domain.lagrange_from_vec(modified_a)
    }
}

impl BasisOps for ExtendedLagrangeCoeff {
    fn constant_term<F: FieldExt>(domain: &EvaluationDomain<F>, scalar: F) -> Polynomial<F, Self> {
        domain.constant_extended(scalar)
    }

    fn linear_term<F: FieldExt>(domain: &EvaluationDomain<F>, scalar: F) -> Polynomial<F, Self> {
        todo!()
    }

    fn rotate<F: FieldExt>(
        domain: &EvaluationDomain<F>,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Polynomial<F, Self> {
        domain.rotate_extended(poly, rotation)
    }

    fn mul<F: FieldExt>(
        _: &EvaluationDomain<F>,
        a: Polynomial<F, Self>,
        b: Polynomial<F, Self>,
    ) -> Polynomial<F, Self> {
        a * &b
    }
}
