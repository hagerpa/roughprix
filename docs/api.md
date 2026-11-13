# API reference

## Overview

--8<-- "README.md:api"

## Model

::: roughprix.RoughHestonParam

::: roughprix.RoughHeston
    options:
      members:
        - eur_call_put
        - charfun
        - log_charfun
        - expansion_state
        - log_charfun_from_field

::: roughprix.heston.HestonExpansionState

## Riccati solutions in closed form

::: roughprix.heston.rough_heston_root_pade_f0

::: roughprix.heston.rough_heston_pade_f0

::: roughprix.heston.h_pade22

::: roughprix.heston.h_pade33

::: roughprix.heston.h_pade44

## Time grids and Volterra operators

::: roughprix.heston.TimeGrid

::: roughprix.heston.Field

::: roughprix.heston.Integrate

::: roughprix.heston.LinearVolterraSolver

::: roughprix.heston.ProductIntegrationRiccatiSolver

::: roughprix.heston.AdamsFractionalRiccatiSolver

::: roughprix.heston.suggested_time_grid_power

## Fourier pricing

::: roughprix.fourier
    options:
      show_root_heading: false
      members:
        - PanelQuadrature
        - build_panel_quadrature
        - fourier_payoff_call_put
        - measure_weights_from_phi
        - eur_call_put_from_weights
        - damped_contour_residues
        - lewis_weights_from_phi
        - eur_call_from_lewis_weights
        - otm_prices_to_ivs
        - adaptive_check_values
        - adaptive_converged
