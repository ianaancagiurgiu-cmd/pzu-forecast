"""Reantreneaza modelul pe date REALE si il salveaza in DB (registrul de modele).
Echivalent cu `python -m pzu.pipeline retrain` dar fara descarcare (foloseste ce e deja in DB).

  python scripts/train.py --window-days 60 --calib-days 90
"""
import argparse

from pzu import db, pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=60)
    ap.add_argument("--calib-days", type=int, default=90)
    ap.add_argument("--version")
    a = ap.parse_args()
    eng = db.get_engine()
    db.init_db(eng)
    print(pipeline.retrain(eng, None, a.window_days, a.calib_days, a.version))


if __name__ == "__main__":
    main()
