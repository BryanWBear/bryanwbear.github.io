// Math & physics articles. LaTeX via KaTeX (protected from marked).

export const ARTICLES = [
  {
    id: 'about-me',
    title: 'Scroll of Bio',
    category: 'about',
    icon: 'map',
    sprite: { col: 11, row: 0 },
    drops: { orange: 0.06, green: 0.06, blue: 0.22, dark: 0.28 },
    content: `
# Bryan Wang

Data scientist and ML engineer. San Francisco Bay Area.

## Work

**Informatica** — current. 5+ years in data science and machine learning.

## Education

UC Berkeley

## Skills

Python, SQL, machine learning, data engineering, statistics

## Contact

[LinkedIn](https://www.linkedin.com/in/bryanmwang)
    `
  },

  {
    id: 'euler-identity',
    title: "Euler's Obsidian Tablet",
    category: 'tech',
    icon: 'scroll',
    sprite: { col: 5, row: 0 },
    drops: { orange: 0.08, green: 0.10, blue: 0.20, dark: 0.30 },
    content: `
# Euler's Identity

$$e^{i\\pi} + 1 = 0$$

Five constants — $e$, $i$, $\\pi$, $1$, $0$ — in one equation.

## Derivation

Euler's formula: for any real $\\theta$,

$$e^{i\\theta} = \\cos\\theta + i\\sin\\theta$$

At $\\theta = \\pi$: $\\cos\\pi = -1$, $\\sin\\pi = 0$, so $e^{i\\pi} = -1$.

## Geometry

The complex exponential $e^{i\\theta}$ traces the unit circle. As $\\theta$ runs from $0$ to $2\\pi$, it completes one full loop. The identity is the halfway point.

<svg width="200" height="200" viewBox="-110 -110 220 220" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:16px auto">
  <circle cx="0" cy="0" r="80" fill="none" stroke="#ffd04055" stroke-width="1"/>
  <line x1="-90" y1="0" x2="90" y2="0" stroke="#8a6a28" stroke-width="1"/>
  <line x1="0" y1="-90" x2="0" y2="90" stroke="#8a6a28" stroke-width="1"/>
  <circle cx="80" cy="0" r="4" fill="#ffd040"/>
  <circle cx="-80" cy="0" r="5" fill="#ff6644"/>
  <path d="M 80 0 A 80 80 0 0 0 -80 0" fill="none" stroke="#88ddff" stroke-width="2" stroke-dasharray="6 3"/>
  <text x="84" y="5" fill="#ffd040" font-size="11" font-family="serif">1</text>
  <text x="-100" y="5" fill="#ff6644" font-size="11" font-family="serif">−1</text>
  <text x="5" y="-84" fill="#eecf78" font-size="10" font-family="serif">i</text>
  <text x="5" y="96" fill="#eecf78" font-size="10" font-family="serif">−i</text>
  <text x="-30" y="-40" fill="#88ddff" font-size="10" font-family="serif">e^iθ</text>
</svg>

---

*Five constants, one truth.*
    `
  },

  {
    id: 'maxwells-equations',
    title: "Maxwell's Gilded Codex",
    category: 'tech',
    icon: 'scroll',
    sprite: { col: 6, row: 0 },
    drops: { orange: 0.06, green: 0.12, blue: 0.22, dark: 0.18 },
    content: `
# Maxwell's Equations

$$\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\varepsilon_0}$$

$$\\nabla \\cdot \\mathbf{B} = 0$$

$$\\nabla \\times \\mathbf{E} = -\\frac{\\partial \\mathbf{B}}{\\partial t}$$

$$\\nabla \\times \\mathbf{B} = \\mu_0 \\mathbf{J} + \\mu_0 \\varepsilon_0 \\frac{\\partial \\mathbf{E}}{\\partial t}$$

## Reading Them

1. Electric charges source diverging $\\mathbf{E}$ fields.
2. No magnetic monopoles — $\\mathbf{B}$ lines are always closed loops.
3. Changing $\\mathbf{B}$ induces curling $\\mathbf{E}$ (Faraday).
4. Currents and changing $\\mathbf{E}$ create curling $\\mathbf{B}$ (Ampère-Maxwell).

## Light Emerges

In vacuum ($\\rho = 0$, $\\mathbf{J} = 0$), equations 3 and 4 combine to give:

$$\\nabla^2 \\mathbf{E} = \\mu_0 \\varepsilon_0 \\frac{\\partial^2 \\mathbf{E}}{\\partial t^2}$$

This is a wave equation. The speed: $c = 1/\\sqrt{\\mu_0 \\varepsilon_0}$, which matches the measured speed of light. Maxwell recognized this in 1865.

---

*Four equations. All of classical electromagnetism.*
    `
  },

  {
    id: 'schrodinger',
    title: "Schrödinger's Cursed Scroll",
    category: 'tech',
    icon: 'scroll',
    sprite: { col: 7, row: 0 },
    drops: { orange: 0.05, green: 0.08, blue: 0.18, dark: 0.25 },
    content: `
# The Schrödinger Equation

$$i\\hbar \\frac{\\partial \\Psi}{\\partial t} = \\hat{H}\\Psi$$

where the Hamiltonian is:

$$\\hat{H} = -\\frac{\\hbar^2}{2m}\\nabla^2 + V(\\mathbf{r}, t)$$

## The Wave Function

$\\Psi(\\mathbf{r}, t)$ is complex-valued. Its modulus squared is a probability density:

$$P(\\mathbf{r}, t) = |\\Psi(\\mathbf{r}, t)|^2$$

## Stationary States

For time-independent potentials, separating variables gives $\\Psi = \\psi(\\mathbf{r}) e^{-iEt/\\hbar}$ where $\\psi$ satisfies:

$$\\hat{H}\\psi = E\\psi$$

This eigenvalue equation produces discrete energy levels — the origin of quantization.

## Strangeness

Before measurement, the system is a superposition of all eigenstates. Measurement collapses the wave function. The mechanism remains philosophically contested.

---

*The mushroom may or may not have dropped this.*
    `
  },

  {
    id: 'stokes-theorem',
    title: "Stokes' Emerald Manifold",
    category: 'tech',
    icon: 'scroll',
    sprite: { col: 9, row: 0 },
    drops: { orange: 0.07, green: 0.15, blue: 0.15, dark: 0.12 },
    content: `
# Generalized Stokes' Theorem

$$\\int_{\\partial \\Omega} \\omega = \\int_{\\Omega} d\\omega$$

A single equation unifying three classical theorems.

## Special Cases in $\\mathbb{R}^3$

**Gradient theorem:**
$$\\int_a^b \\nabla f \\cdot d\\mathbf{r} = f(b) - f(a)$$

**Classical Stokes:**
$$\\iint_S (\\nabla \\times \\mathbf{F}) \\cdot d\\mathbf{S} = \\oint_{\\partial S} \\mathbf{F} \\cdot d\\mathbf{r}$$

**Divergence theorem:**
$$\\iiint_V (\\nabla \\cdot \\mathbf{F})\\, dV = \\oiint_{\\partial V} \\mathbf{F} \\cdot d\\mathbf{S}$$

## The Abstract Structure

The operators $\\partial$ (boundary) and $d$ (exterior derivative) are adjoint: $\\partial^2 = 0$ and $d^2 = 0$. This duality is the foundation of de Rham cohomology, and ultimately Hodge theory.

---

*Boundaries and interiors. Topology made analytic.*
    `
  },

  {
    id: 'noether-theorem',
    title: "Noether's Eternal Ring",
    category: 'tech',
    icon: 'scroll',
    sprite: { col: 10, row: 0 },
    drops: { orange: 0.05, green: 0.09, blue: 0.16, dark: 0.22 },
    content: `
# Noether's Theorem

Every continuous symmetry of a physical system corresponds to a conserved quantity.

## Formal Statement

If the action $S = \\int \\mathcal{L}(q, \\dot{q}, t)\\, dt$ is invariant under $q \\to q + \\varepsilon \\delta q$, then:

$$J = \\frac{\\partial \\mathcal{L}}{\\partial \\dot{q}} \\delta q \\quad \\text{satisfies} \\quad \\frac{dJ}{dt} = 0$$

## The Three Big Correspondences

| Symmetry | Conserved Quantity |
|----------|-------------------|
| Time translation | Energy $E$ |
| Spatial translation | Momentum $\\mathbf{p}$ |
| Rotation | Angular momentum $\\mathbf{L}$ |

Conservation of energy is not an axiom. It is a *consequence* of the fact that the laws of physics are the same today as yesterday.

---

*Proved in 1915. Einstein said she was the most significant creative mathematical genius yet produced.*
    `
  },

  {
    id: 'cauchy-schwarz',
    title: "Cauchy's Iron Inequality",
    category: 'tech',
    icon: 'scroll',
    sprite: { col: 4, row: 0 },
    drops: { orange: 0.10, green: 0.14, blue: 0.12, dark: 0.16 },
    content: `
# The Cauchy-Schwarz Inequality

For any vectors $\\mathbf{u}, \\mathbf{v}$ in an inner product space:

$$|\\langle \\mathbf{u}, \\mathbf{v} \\rangle|^2 \\leq \\langle \\mathbf{u}, \\mathbf{u} \\rangle \\cdot \\langle \\mathbf{v}, \\mathbf{v} \\rangle$$

Equality iff $\\mathbf{u}$ and $\\mathbf{v}$ are linearly dependent.

## Proof

Consider $f(t) = \\|\\mathbf{u} + t\\mathbf{v}\\|^2 \\geq 0$:

$$f(t) = \\|\\mathbf{u}\\|^2 + 2t\\langle\\mathbf{u},\\mathbf{v}\\rangle + t^2\\|\\mathbf{v}\\|^2 \\geq 0$$

Non-negative quadratic in $t$ $\\Rightarrow$ discriminant $\\leq 0$:

$$4\\langle\\mathbf{u},\\mathbf{v}\\rangle^2 - 4\\|\\mathbf{u}\\|^2 \\|\\mathbf{v}\\|^2 \\leq 0 \\qquad \\square$$

## Instances

- **Probability:** $\\operatorname{Cov}(X,Y)^2 \\leq \\operatorname{Var}(X)\\operatorname{Var}(Y)$
- **Integrals:** $\\left(\\int fg\\right)^2 \\leq \\int f^2 \\int g^2$
- **Quantum:** $\\sigma_x \\sigma_p \\geq \\hbar/2$ (Heisenberg)

---

*The most used inequality in mathematics.*
    `
  },

  {
    id: 'riemann-roch',
    title: "Riemann-Roch Cipher Stone",
    category: 'tech',
    icon: 'scroll',
    sprite: { col: 12, row: 0 },
    drops: { orange: 0.04, green: 0.08, blue: 0.18, dark: 0.28 },
    content: `
# Riemann-Roch Theorem

For a divisor $D$ on a compact Riemann surface $C$ of genus $g$:

$$\\ell(D) - \\ell(K_C - D) = \\deg D - g + 1$$

$\\ell(D) = \\dim H^0(C, \\mathcal{O}(D))$ counts independent meromorphic functions with poles bounded by $D$. $K_C$ is the canonical divisor.

## What It Computes

Riemann-Roch answers: *how many meromorphic functions on a surface can have prescribed poles?* The answer depends only on the degree of the divisor and the topology (genus) of the surface.

## The Torus

<svg width="240" height="140" viewBox="-120 -70 240 140" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:16px auto">
  <ellipse cx="0" cy="0" rx="100" ry="58" fill="none" stroke="#ffd040" stroke-width="2"/>
  <ellipse cx="0" cy="0" rx="38" ry="20" fill="#160c04" stroke="#ffd040" stroke-width="1.5"/>
  <path d="M -100 0 Q -60 -18 0 -18 Q 60 -18 100 0" fill="none" stroke="#c8981e" stroke-width="1.5" stroke-dasharray="5 3"/>
  <path d="M -100 0 Q -60 18 0 18 Q 60 18 100 0" fill="none" stroke="#c8981e" stroke-width="1.5"/>
  <path d="M -38 0 Q -20 -12 0 -12 Q 20 -12 38 0" fill="none" stroke="#88ddff" stroke-width="1" stroke-dasharray="3 2"/>
  <path d="M -38 0 Q -20 12 0 12 Q 20 12 38 0" fill="none" stroke="#88ddff" stroke-width="1"/>
  <text x="-8" y="-52" fill="#eecf78" font-size="11" font-family="serif" font-style="italic">genus 1</text>
</svg>

For a torus ($g=1$): $\\ell(D) - \\ell(K_C - D) = \\deg D$.

## Higher Dimensions

The Hirzebruch-Riemann-Roch theorem extends this to algebraic varieties of any dimension, relating the Euler characteristic of a vector bundle to characteristic classes. Grothendieck's version works for morphisms between schemes.

---

*Algebraic geometry's most powerful computational tool.*
    `
  },

  {
    id: 'cobordism',
    title: "Thom's Cobordism Crystal",
    category: 'tech',
    icon: 'scroll',
    sprite: { col: 13, row: 0 },
    drops: { orange: 0.03, green: 0.07, blue: 0.15, dark: 0.30 },
    content: `
# Cobordism

Two closed $n$-manifolds $M_0$ and $M_1$ are **cobordant** if there exists a compact $(n+1)$-manifold $W$ with:

$$\\partial W = M_0 \\sqcup M_1$$

<svg width="320" height="130" viewBox="0 0 320 130" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:16px auto">
  <ellipse cx="38" cy="40" rx="16" ry="32" fill="none" stroke="#ffd040" stroke-width="2"/>
  <ellipse cx="38" cy="95" rx="16" ry="22" fill="none" stroke="#ffd040" stroke-width="2"/>
  <ellipse cx="282" cy="65" rx="16" ry="50" fill="none" stroke="#ff6644" stroke-width="2"/>
  <path d="M 22 8 Q 80 10 150 30 Q 220 50 266 15" fill="none" stroke="#88ddff" stroke-width="2"/>
  <path d="M 54 8 Q 100 5 150 30 Q 210 52 266 15" fill="none" stroke="#88ddff" stroke-width="1" stroke-dasharray="4 2"/>
  <path d="M 22 72 Q 70 68 150 65 Q 220 62 266 115" fill="none" stroke="#88ddff" stroke-width="2"/>
  <path d="M 54 72 Q 90 65 150 65 Q 215 64 266 115" fill="none" stroke="#88ddff" stroke-width="1" stroke-dasharray="4 2"/>
  <path d="M 22 117 Q 80 120 150 100 Q 220 80 266 115" fill="none" stroke="#88ddff" stroke-width="2"/>
  <path d="M 54 117 Q 90 118 150 100 Q 215 82 266 115" fill="none" stroke="#88ddff" stroke-width="1" stroke-dasharray="4 2"/>
  <text x="20" y="130" fill="#ffd040" font-size="10" font-family="serif" font-style="italic">M₀</text>
  <text x="262" y="128" fill="#ff6644" font-size="10" font-family="serif" font-style="italic">M₁</text>
  <text x="135" y="15" fill="#88ddff" font-size="10" font-family="serif" font-style="italic">W</text>
</svg>

## Thom's Theorem

René Thom (1954) computed the unoriented cobordism ring $\\mathfrak{N}_*$. For oriented cobordism:

$$\\Omega_*^{\\mathrm{SO}} \\otimes \\mathbb{Q} \\cong \\mathbb{Q}[y_4, y_8, y_{12}, \\ldots]$$

generated by $\\mathbb{CP}^{2k}$ in dimension $4k$. Two oriented manifolds are rationally cobordant iff they have the same Pontryagin numbers.

## Why It Matters

Cobordism is a fundamental equivalence relation on manifolds. The cobordism groups classify manifolds up to this relation. Thom's method — the Thom isomorphism and Pontryagin-Thom construction — became a template for generalized cohomology theories.

Atiyah showed that cobordism is a generalized cohomology theory, and the study of cobordism categories now connects directly to topological quantum field theory via the **cobordism hypothesis** (Baez-Dolan, proved by Lurie).

---

*Two manifolds, one interpolating space, one theorem.*
    `
  },

  {
    id: 'quadratic-reciprocity',
    title: "Gauss's Bronze Reciprocity Law",
    category: 'tech',
    icon: 'scroll',
    sprite: { col: 14, row: 0 },
    drops: { orange: 0.09, green: 0.13, blue: 0.14, dark: 0.18 },
    content: `
# Quadratic Reciprocity

For distinct odd primes $p$ and $q$:

$$\\left(\\frac{p}{q}\\right)\\left(\\frac{q}{p}\\right) = (-1)^{\\frac{p-1}{2}\\cdot\\frac{q-1}{2}}$$

where $\\left(\\frac{a}{p}\\right)$ is the Legendre symbol: $+1$ if $a$ is a square mod $p$, $-1$ if not.

## Reading It

The sign on the right depends on whether $p$ or $q$ are $\\equiv 3 \\pmod{4}$:

- If both $p \\equiv q \\equiv 3 \\pmod{4}$: the symbols are **negatives** of each other.
- Otherwise: the symbols are **equal**.

So knowing whether $p$ is a square mod $q$ tells you whether $q$ is a square mod $p$ — with at most a sign flip.

## Example

Is $3$ a square mod $11$? By reciprocity:

$$\\left(\\frac{3}{11}\\right) = \\left(\\frac{11}{3}\\right) = \\left(\\frac{2}{3}\\right) = -1$$

So $3$ is not a quadratic residue mod $11$.

## Gauss's Gem

Gauss called this the *theorema aureum* — the golden theorem — and gave 8 independent proofs. It is the prototype for class field theory and the Artin reciprocity law.

---

*The first deep theorem in number theory. Gauss proved it at 19.*
    `
  },

  {
    id: 'prime-number-theorem',
    title: "Prime Number Theorem Shard",
    category: 'tech',
    icon: 'scroll',
    sprite: { col: 15, row: 0 },
    drops: { orange: 0.10, green: 0.12, blue: 0.12, dark: 0.14 },
    content: `
# Prime Number Theorem

$$\\pi(x) \\sim \\frac{x}{\\ln x} \\quad \\text{as } x \\to \\infty$$

$\\pi(x)$ counts primes up to $x$. The logarithmic integral gives a sharper approximation:

$$\\pi(x) = \\operatorname{Li}(x) + O\\!\\left(x\\, e^{-c\\sqrt{\\ln x}}\\right), \\quad \\operatorname{Li}(x) = \\int_2^x \\frac{dt}{\\ln t}$$

## Ulam Spiral

Integers arranged in a spiral, primes highlighted — they cluster on diagonals.

<svg width="210" height="210" viewBox="0 0 210 210" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:16px auto;background:#0a0604">
  <rect width="210" height="210" fill="#0a0604"/>
  <!-- Ulam spiral 11x11, primes marked gold -->
  <!-- Center=105,105, cell=18px -->
  <!-- Row/col offsets for spiral: precomputed prime positions -->
  <!-- Primes up to 121: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113 -->
  <!-- Spiral: n=1 at center, going right then up then left then down -->
  <g fill="#2a1810">
    <!-- all cells background -->
    <rect x="6" y="6" width="198" height="198" fill="none"/>
  </g>
  <!-- Mark every cell as small rect, primes in gold -->
  <!-- Spiral order starting at center (5,5) in 0-indexed 11x11 grid -->
  <!-- I'll just mark prime positions directly -->
  <rect x="96" y="96" width="18" height="18" fill="#1a1208" rx="1"/> <!-- 1: not prime -->
  <rect x="114" y="96" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 2 prime -->
  <rect x="114" y="78" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 3 prime -->
  <rect x="96" y="78" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="78" y="78" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 5 -->
  <rect x="78" y="96" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="78" y="114" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 7 -->
  <rect x="96" y="114" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="114" y="114" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="132" y="114" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="132" y="96" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 11 -->
  <rect x="132" y="78" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 13 -->
  <rect x="132" y="60" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="114" y="60" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="96" y="60" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="78" y="60" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="60" y="60" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="60" y="78" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="60" y="96" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 17 -->
  <rect x="60" y="114" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 19 -->
  <rect x="60" y="132" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="78" y="132" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="96" y="132" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="114" y="132" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="132" y="132" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="150" y="132" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="150" y="114" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="150" y="96" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="150" y="78" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="150" y="60" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 23 -->
  <rect x="150" y="42" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="132" y="42" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="114" y="42" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="96" y="42" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="78" y="42" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="60" y="42" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="42" y="42" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 29 -->
  <rect x="42" y="60" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 31 -->
  <rect x="42" y="78" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="42" y="96" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="42" y="114" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="42" y="132" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="42" y="150" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="60" y="150" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="78" y="150" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="96" y="150" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 37 -->
  <rect x="114" y="150" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="132" y="150" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="150" y="150" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="168" y="150" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="168" y="132" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="168" y="114" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="168" y="96" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="168" y="78" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="168" y="60" width="18" height="18" fill="#1a1208" rx="1"/>
  <rect x="168" y="42" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 41 -->
  <rect x="168" y="24" width="18" height="18" fill="#ffd040" rx="1" opacity="0.9"/> <!-- 43 -->
</svg>

*Gold = prime. Diagonals emerge.*

## The Zeta Connection

Riemann showed $\\pi(x)$ is governed by the zeros of:

$$\\zeta(s) = \\sum_{n=1}^{\\infty} \\frac{1}{n^s} = \\prod_{p \\text{ prime}} \\frac{1}{1-p^{-s}}$$

The PNT follows from the fact that $\\zeta(s) \\neq 0$ on the line $\\operatorname{Re}(s) = 1$.

---

*Primes thin out, but never stop. The rate is logarithmic.*
    `
  },

  {
    id: 'weil-conjectures',
    title: "Weil's Crystalline Tablet",
    category: 'tech',
    icon: 'scroll',
    sprite: { col: 16, row: 0 },
    drops: { orange: 0.03, green: 0.06, blue: 0.14, dark: 0.25 },
    content: `
# The Weil Conjectures

For a smooth projective variety $X$ over $\\mathbb{F}_q$, define the zeta function:

$$Z(X, T) = \\exp\\!\\left(\\sum_{n=1}^{\\infty} |X(\\mathbb{F}_{q^n})| \\frac{T^n}{n}\\right)$$

## The Four Conjectures (Weil, 1949)

**1. Rationality.** $Z(X, T) \\in \\mathbb{Q}(T)$ is a rational function.

**2. Functional equation.** With $q = p^a$ and $d = \\dim X$:

$$Z\\!\\left(X, \\frac{1}{q^d T}\\right) = \\pm\\, q^{d\\chi/2} T^\\chi Z(X, T)$$

where $\\chi$ is the Euler characteristic.

**3. Riemann hypothesis.** The zeros and poles of $Z(X,T)$ have $|\\alpha| = q^{i/2}$ for integer $i$.

**4. Betti numbers.** The degrees of the numerator/denominator factors match the Betti numbers of the complex manifold $X(\\mathbb{C})$.

## Proofs

- Conjectures 1 and 4: Dwork (1960), Grothendieck (1965) via étale cohomology
- The Riemann hypothesis: **Deligne (1974)** — Fields Medal

Grothendieck built étale cohomology specifically to prove these. The machinery required: schemes, sites, topoi, $\\ell$-adic sheaves. A decade of algebraic geometry as infrastructure for one theorem.

---

*Counting points over finite fields. The answer is a zeta function.*
    `
  },
];

// ── DROP TABLE ────────────────────────────────────────────────────────────────

const COIN_TABLE = {
  orange: [['bronze', 70], ['gold', 93], ['bill', 99], ['meso', 100]],
  green:  [['bronze', 60], ['gold', 88], ['bill', 97], ['meso', 100]],
  blue:   [['bronze', 45], ['gold', 80], ['bill', 95], ['meso', 100]],
  dark:   [['bronze', 28], ['gold', 65], ['bill', 88], ['meso', 100]],
};

export const COIN_AMOUNTS = {
  bronze: [10, 50],
  gold:   [100, 500],
  bill:   [800, 3000],
  meso:   [2000, 8000],
};

function _rollCoinTier(mushroomType) {
  const table = COIN_TABLE[mushroomType] || COIN_TABLE.orange;
  const roll = Math.random() * 100;
  for (const [tier, cum] of table) {
    if (roll < cum) return tier;
  }
  return 'bronze';
}

export function rollDrops(mushroomType) {
  const articles = ARTICLES
    .filter(a => a.drops[mushroomType] && Math.random() < a.drops[mushroomType])
    .map(a => a.id);
  const coinCount = Math.random() < 0.40 ? 2 : 1;
  const coins = Array.from({ length: coinCount }, () => _rollCoinTier(mushroomType));
  return { articles, coins };
}

export function getArticle(id) {
  return ARTICLES.find(a => a.id === id) || null;
}

export function getDropsForType(mushroomType) {
  return ARTICLES.filter(a => a.drops[mushroomType] > 0);
}
