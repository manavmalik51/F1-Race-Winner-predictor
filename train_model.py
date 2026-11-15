"""Offline general-season training; Streamlit only loads the resulting PKL."""
import argparse, pickle, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import f1_race_pipeline as f1

OUT=Path(__file__).resolve().parent; ROOT=OUT.parent; DEFAULT=ROOT/'work'/'f1_history_source.csv'

def main():
 p=argparse.ArgumentParser(); p.add_argument('--refresh',action='store_true'); p.add_argument('--folds',type=int,default=5); p.add_argument('--csv-dir',type=Path,help='optional Kaggle directory with races/results/qualifying CSVs'); a=p.parse_args()
 if a.csv_dir: raw=f1.load_kaggle_csvs(a.csv_dir)
 else:
  if a.refresh or not DEFAULT.exists(): subprocess.run([sys.executable,str(OUT/'fetch_public_f1.py')],cwd=ROOT,check=True)
  raw=pd.read_csv(DEFAULT)
 print(f'History: {len(raw):,} driver-race rows, {raw.race_id.nunique()} races, seasons {sorted(raw.year.dropna().unique().astype(int))}')
 engineered=f1.engineer_features(raw); scores,best=f1.compare_models(engineered,a.folds); scores.to_csv(OUT/'model_comparison.csv',index=False)
 print('\nGrouped cross-validation (GroupKFold by race_id):\n'+scores.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
 models,meta=f1.fit_best_model(engineered,best)
 meta.update({'trained_at_utc':datetime.now(timezone.utc).isoformat(),'history_years':sorted(raw.year.dropna().unique().astype(int).tolist()),'cv_scores':scores.to_dict(orient='records'),'source':'Jolpica + OpenF1 + public FastF1-derived weather'})
 bundle=f1.make_bundle(models,meta,raw)
 with (OUT/'f1_all_races_model.pkl').open('wb') as h: pickle.dump(bundle,h,protocol=pickle.HIGHEST_PROTOCOL)
 print(f"\nSelected {best}; saved {OUT/'f1_all_races_model.pkl'}")
if __name__=='__main__': main()
