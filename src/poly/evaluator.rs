use std::{
    collections::HashMap,
    hash::Hash,
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
pub(crate) struct AstLeaf<E, Id, B: Basis> {
    id: Id,
    rotation: Rotation,
    _evaluator: PhantomData<(E, B)>,
}

impl<E, Id: Copy, B: Basis> AstLeaf<E, Id, B> {
    /// Produces a new `AstLeaf` node corresponding to the underlying polynomial at a
    /// _new_ rotation. Existing rotations applied to this leaf node are ignored and the
    /// returned polynomial is not rotated _relative_ to the previous structure.
    pub(crate) fn with_rotation(&self, rotation: Rotation) -> Self {
        AstLeaf {
            id: self.id,
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
pub(crate) struct Evaluator<E, Id: Eq + Hash, F: Field, B: Basis> {
    polys: HashMap<Id, Polynomial<F, B>>,
    _context: E,
}

/// Constructs a new `Evaluator`.
///
/// The `context` parameter is used to provide type safety for evaluators. It ensures that
/// an evaluator will only be used to evaluate [`Ast`]s containing [`AstLeaf`]s obtained
/// from itself. It should be set to the empty closure `|| {}`, because anonymous closures
/// all have unique types.
pub(crate) fn new_evaluator<E: Fn() + Clone, Id: Eq + Hash, F: Field, B: Basis>(
    context: E,
) -> Evaluator<E, Id, F, B> {
    Evaluator {
        polys: HashMap::default(),
        _context: context,
    }
}

impl<E, F: Field, B: Basis> Evaluator<E, usize, F, B> {
    /// Registers the given polynomial for use in this evaluation context.
    ///
    /// This API treats each registered polynomial as unique, even if the same polynomial
    /// is added multiple times.
    pub(crate) fn register_poly(&mut self, poly: Polynomial<F, B>) -> AstLeaf<E, usize, B> {
        let id = self.polys.len();
        self.polys.insert(id, poly);

        AstLeaf {
            id,
            rotation: Rotation::cur(),
            _evaluator: PhantomData::default(),
        }
    }
}

impl<E, Id: Copy + Eq + Hash, F: Field, B: Basis> Evaluator<E, Id, F, B> {
    /// Registers the given polynomial for use in this evaluation context.
    ///
    /// This API registers the polynomial with the given identifier, as a cheaper way of
    /// detecting duplicate polynomials than an equality check. The caller must ensure the
    /// one-to-one mapping between identifiers and polynomials.
    pub(crate) fn register_poly_as(
        &mut self,
        id: Id,
        poly: &Polynomial<F, B>,
    ) -> AstLeaf<E, Id, B> {
        self.polys.entry(id).or_insert_with(|| poly.clone());

        AstLeaf {
            id,
            rotation: Rotation::cur(),
            _evaluator: PhantomData::default(),
        }
    }

    /// Evaluates the given polynomial operation against this context.
    pub(crate) fn evaluate(
        &self,
        ast: &Ast<E, Id, F, B>,
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, B>
    where
        F: FieldExt,
        B: BasisOps,
    {
        match ast {
            Ast::Poly(AstLeaf { id, rotation, .. }) => {
                B::rotate(domain, self.polys.get(id).unwrap(), *rotation)
            }
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
pub(crate) struct AstMul<E, Id, F: Field, B: Basis>(Box<Ast<E, Id, F, B>>, Box<Ast<E, Id, F, B>>);

/// A polynomial operation backed by an [`Evaluator`].
#[derive(Clone)]
pub(crate) enum Ast<E, Id, F: Field, B: Basis> {
    Poly(AstLeaf<E, Id, B>),
    Add(Box<Ast<E, Id, F, B>>, Box<Ast<E, Id, F, B>>),
    Mul(AstMul<E, Id, F, B>),
    Scale(Box<Ast<E, Id, F, B>>, F),
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

impl<E, Id, F: Field, B: Basis> From<AstLeaf<E, Id, B>> for Ast<E, Id, F, B> {
    fn from(leaf: AstLeaf<E, Id, B>) -> Self {
        Ast::Poly(leaf)
    }
}

impl<E, Id, F: Field, B: Basis> Neg for Ast<E, Id, F, B> {
    type Output = Ast<E, Id, F, B>;

    fn neg(self) -> Self::Output {
        Ast::Scale(Box::new(self), -F::one())
    }
}

impl<E: Clone, Id: Clone, F: Field, B: Basis> Neg for &Ast<E, Id, F, B> {
    type Output = Ast<E, Id, F, B>;

    fn neg(self) -> Self::Output {
        -(self.clone())
    }
}

impl<E, Id, F: Field, B: Basis> Add<Ast<E, Id, F, B>> for Ast<E, Id, F, B> {
    type Output = Ast<E, Id, F, B>;

    fn add(self, other: Ast<E, Id, F, B>) -> Self::Output {
        Ast::Add(Box::new(self), Box::new(other))
    }
}

impl<'a, E: Clone, Id: Clone, F: Field, B: Basis> Add<&'a Ast<E, Id, F, B>>
    for &'a Ast<E, Id, F, B>
{
    type Output = Ast<E, Id, F, B>;

    fn add(self, other: &'a Ast<E, Id, F, B>) -> Self::Output {
        self.clone() + other.clone()
    }
}

impl<E, Id, F: Field, B: Basis> Sub<Ast<E, Id, F, B>> for Ast<E, Id, F, B> {
    type Output = Ast<E, Id, F, B>;

    fn sub(self, other: Ast<E, Id, F, B>) -> Self::Output {
        self + (-other)
    }
}

impl<'a, E: Clone, Id: Clone, F: Field, B: Basis> Sub<&'a Ast<E, Id, F, B>>
    for &'a Ast<E, Id, F, B>
{
    type Output = Ast<E, Id, F, B>;

    fn sub(self, other: &'a Ast<E, Id, F, B>) -> Self::Output {
        self + &(-other)
    }
}

impl<E, Id, F: Field> Mul<Ast<E, Id, F, LagrangeCoeff>> for Ast<E, Id, F, LagrangeCoeff> {
    type Output = Ast<E, Id, F, LagrangeCoeff>;

    fn mul(self, other: Ast<E, Id, F, LagrangeCoeff>) -> Self::Output {
        Ast::Mul(AstMul(Box::new(self), Box::new(other)))
    }
}

impl<'a, E: Clone, Id: Clone, F: Field> Mul<&'a Ast<E, Id, F, LagrangeCoeff>>
    for &'a Ast<E, Id, F, LagrangeCoeff>
{
    type Output = Ast<E, Id, F, LagrangeCoeff>;

    fn mul(self, other: &'a Ast<E, Id, F, LagrangeCoeff>) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<E, Id, F: Field> Mul<Ast<E, Id, F, ExtendedLagrangeCoeff>>
    for Ast<E, Id, F, ExtendedLagrangeCoeff>
{
    type Output = Ast<E, Id, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: Ast<E, Id, F, ExtendedLagrangeCoeff>) -> Self::Output {
        Ast::Mul(AstMul(Box::new(self), Box::new(other)))
    }
}

impl<'a, E: Clone, Id: Clone, F: Field> Mul<&'a Ast<E, Id, F, ExtendedLagrangeCoeff>>
    for &'a Ast<E, Id, F, ExtendedLagrangeCoeff>
{
    type Output = Ast<E, Id, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: &'a Ast<E, Id, F, ExtendedLagrangeCoeff>) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<E, Id, F: Field, B: Basis> Mul<F> for Ast<E, Id, F, B> {
    type Output = Ast<E, Id, F, B>;

    fn mul(self, other: F) -> Self::Output {
        Ast::Scale(Box::new(self), other)
    }
}

impl<E: Clone, Id: Clone, F: Field, B: Basis> Mul<F> for &Ast<E, Id, F, B> {
    type Output = Ast<E, Id, F, B>;

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
