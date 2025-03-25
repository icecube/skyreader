"""An example script to make a plot from SkyDriver's serialized result."""


import argparse

from rest_tools.client import RestClient, SavedDeviceGrantAuth
from skyreader import SkyScanResult
from skyreader.plot import SkyScanPlotter


def get_rest_client() -> RestClient:
    """Get REST client for talking to SkyDriver.

    This will present a QR code in the terminal for initial validation.
    """

    # NOTE: If your script will not be interactive (like a cron job),
    # then you need to first run your script manually to validate using
    # the QR code in the terminal.

    return SavedDeviceGrantAuth(
        "https://skydriver.icecube.aq",
        token_url="https://keycloak.icecube.wisc.edu/auth/realms/IceCube",
        filename="device-refresh.token",
        client_id="skydriver-external",
        retries=0,
    )


def main() -> None:
    """Make plots."""
    parser = argparse.ArgumentParser(
        description="Make plot of scan result from SkyDriver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scan-id", help="skydriver scan id", required=True)
    args = parser.parse_args()

    rc = get_rest_client()
    serialized = rc.request_seq("GET", f"/scan/{args.scan_id}/result")["skyscan_result"]

    plotter = SkyScanPlotter()

    result = SkyScanResult.deserialize(serialized)

    plotter.create_plot(result)
    plotter.create_plot_zoomed(result, plot_bounding_box=True)


if __name__ == "__main__":
    main()
