import numpy as np

import cylinder


class Field:
    """
    Parâmetros físicos
    """

    eps_0 = 8.854e-12
    eps_g = eps_0
    eps_d = 5 * eps_0

    r_a = 13.5e-3
    r_d = 14.8e-3
    r_b = 18e-3

    L = 20e-2

    V_0 = 10e3

    geometric_factor = eps_g / (eps_g * np.log(r_d / r_a) + eps_d * np.log(r_b / r_d))

    coords = cylinder.CoaxialCylinder(r_a, r_b, L)

    def calculate_field(self):
        r, z = self.coords.coordinates

        # ------------------------------------------------------------
        # Cálculo de E_r(r,z) e E_z(r, z)
        # ------------------------------------------------------------
        half_L = self.L / 2

        term1 = (z + half_L) / np.sqrt(r**2 + (z + half_L) ** 2)
        term2 = (z - half_L) / np.sqrt(r**2 + (z - half_L) ** 2)

        axial_correction_factor = (term1 - term2) / 2

        Er = self.V_0 * self.geometric_factor / r * axial_correction_factor

        Ez = (
            self.V_0
            * self.geometric_factor
            * (
                1 / np.sqrt(r**2 + (z - half_L) ** 2)
                - 1 / np.sqrt(r**2 + (z + half_L) ** 2)
            )
        )

        return self.coords.to_cartesian(Er, Ez)
