import pyvista as pv

import field


class Plot:
    """
    PyVista plot of the electric field using fixed-size glyphs and magnitude-based coloring.
    """

    def __init__(self, glyph_size=1e-3):
        """
        Initialize the PyVista plotter and field model.

        Parameters
        ----------
        glyph_size : float, optional
            Length of the glyphs representing the electric field vectors.
            The glyph size is fixed and does not scale with field magnitude.
        """

        self.glyph_size = glyph_size
        self.field = field.DielectricField()
        self.plotter = pv.Plotter()

    def show(self):
        """
        Render the electric field visualization.

        The method computes the electric field, creates a point cloud,
        attaches vector and magnitude data, and renders oriented glyphs
        colored by field magnitude.

        Returns
        -------
        None
        """

        points, vectors_unit, mag = self.field.calculate_field()

        cloud = pv.PolyData(points)
        cloud["vectors"] = vectors_unit
        cloud["mag"] = mag

        glyphs = cloud.glyph(orient="vectors", scale=False, factor=self.glyph_size)

        self.plotter.add_mesh(
            glyphs,
            scalars="mag",
            cmap="viridis",
            scalar_bar_args={"title": "|E| [V/m]"},
        )

        AXIS_LENGTH = 101e-3  # L / 2 + delta

        x_axis = pv.Line((-AXIS_LENGTH, 0, 0), (AXIS_LENGTH, 0, 0))
        y_axis = pv.Line((0, -AXIS_LENGTH, 0), (0, AXIS_LENGTH, 0))
        z_axis = pv.Line((0, 0, -AXIS_LENGTH), (0, 0, AXIS_LENGTH))

        LINE_WIDTH = 1

        self.plotter.add_mesh(x_axis, color="red", line_width=LINE_WIDTH, name="x_axis")
        self.plotter.add_mesh(
            y_axis, color="green", line_width=LINE_WIDTH, name="y_axis"
        )
        self.plotter.add_mesh(
            z_axis, color="blue", line_width=LINE_WIDTH, name="z_axis"
        )

        pv.Plotter.add_axes(self.plotter)
        self.plotter.show()
