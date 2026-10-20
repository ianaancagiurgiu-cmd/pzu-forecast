"""OPCOM - optional (150 EUR/luna). In v1 sursa principala de preturi e ENTSO-E (A44).

Conectorul se activeaza doar daca OPCOM_API_KEY este setat; documentatia API-ului
este disponibila doar participantilor, deci implementarea ramane de completat la integrare.
"""
from pzu.config import get_settings


def is_enabled() -> bool:
    return bool(get_settings().opcom_api_key)


def fetch_prices(start, end):
    raise NotImplementedError(
        "Conectorul OPCOM nu e implementat: necesita cont de participant si documentatia API."
    )
