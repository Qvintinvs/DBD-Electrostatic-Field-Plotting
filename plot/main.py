import plot


def main():
    plt = plot.PlotBuilder()

    plt.enable_parallel_projection_y_up()

    plt.add_error_text()
    plt.add_glyphs()
    plt.add_axes()

    plt.auto_zoom()

    plt.show()


if __name__ == "__main__":
    main()
