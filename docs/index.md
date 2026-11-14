---
hide:
  - navigation
  - toc
---

<div class="rp-hero" markdown>

<div class="rp-title" markdown>

![roughprix logo](assets/logo.svg){ .rp-logo }

# roughprix

</div>

European option prices in the rough Heston model, by Fourier inversion, with a Taylor expansion in the Hurst exponent $H$.

[Get started](#installation){ .md-button .md-button--primary }
[Paper](https://arxiv.org/abs/2606.16619){ .md-button }
[GitHub](https://github.com/hagerpa/roughprix){ .md-button }

</div>

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } __Fast Fourier pricing__

    ---

    A smile of nine strikes in about 40 to 80 ms, with JAX-compiled Riccati solvers.

-   :material-sigma:{ .lg .middle } __Expansion in $H$__

    ---

    Price many Hurst exponents from one anchor solve: the coefficients come from linear Volterra equations.

-   :material-approximately-equal:{ .lg .middle } __Closed-form bases__

    ---

    Exact Heston at $H = \tfrac12$, and a tail-matched root-Padé approximation for rough $H$.

-   :material-check-decagram:{ .lg .middle } __Checked accuracy__

    ---

    Default prices within 0.003 bp of 72 reference smiles; a built-in check warns when an approximation is off.

</div>

--8<-- "README.md:install"

--8<-- "README.md:quickstart"

Continue with [Pricing](pricing.md) and the [H-expansion](expansion.md).

--8<-- "README.md:model"

--8<-- "README.md:methods"

--8<-- "README.md:roadmap"
