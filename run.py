import importlib


EXPERIMENT = "lorentznet"

SETTINGS = {
    "mnist": {
        "data_root": "data",
        "output_root": "outputs/mnist",
        "regime": "both",
    },
    "deepsets": {
        "data_root": "src/data/jetid",
        "output_root": "outputs/deepsets",
        "regime": "both",
    },
    "lorentznet": {
        "data_root": "src/data/jetid",
        "output_root": "outputs/lorentznet",
        "regime": "both",
    },
}


def main():
    package = f"{__package__}." if __package__ else ""
    module = importlib.import_module(f"{package}{EXPERIMENT}.run")
    module.main(**SETTINGS[EXPERIMENT])


if __name__ == "__main__":
    main()
