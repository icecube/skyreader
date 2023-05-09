"""An example script to make a plot from a numpy scan result."""


import argparse
from skyreader import SkyScanResult


def main() -> None:
    """Make plots."""
    parser = argparse.ArgumentParser(
        description="Make plot of scan result from scan result file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="filename")
    parser.add_argument("--bounding-box", action="store_true", default=False)
    args = parser.parse_args()

    result = SkyScanResult.read_npz(args.filename)

    result.create_plot(dosave=True)
    result.create_plot_zoomed(dosave=True, plot_bounding_box=args.bounding_box)


if __name__ == "__main__":
    main()
