"""General F1 race predictor: loaders, features, CV selection and event inference."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
from datetime import datetime, timezone
import json, math, re, urllib.request
import numpy as np
import pandas as pd

ALGORITHMS=("LightGBM","XGBoost","ExtraTrees")
FEATURES=["grid_position","qualifying_time_seconds","straight_line_speed_rank","positions_gained_avg",
 "rolling_5_race_points","rolling_5_race_pace_delta","track_temp_c","is_low_drag_cluster",
 "cold_track_warmup_index","driver_experience_races","speed_overtake_strength"]
LOW_DRAG=("monza","baku","jeddah","vegas")

def _slug(x): return "" if pd.isna(x) else re.sub(r"[^a-z0-9]+"," ",str(x).lower()).strip()
def _api(url):
 req=urllib.request.Request(url,headers={"User-Agent":"General-F1-Predictor/1.0"})
 with urllib.request.urlopen(req,timeout=30) as r: return json.loads(r.read())

def load_kaggle_csvs(source: str|Path|Mapping[str,str|Path])->pd.DataFrame:
 """Load Ergast/Kaggle results, races, qualifying, plus optional lookups."""
 paths={Path(k).stem.lower():Path(v) for k,v in source.items()} if isinstance(source,Mapping) else {p.stem.lower():p for p in Path(source).glob('*.csv')}
 miss={'results','races','qualifying'}-paths.keys()
 if miss: raise FileNotFoundError(f'Missing CSVs: {sorted(miss)}')
 races,results,q=(pd.read_csv(paths[k]) for k in ('races','results','qualifying'))
 races=races.rename(columns={'raceId':'race_id','circuitId':'circuit_id','name':'event_name'})
 results=results.rename(columns={'raceId':'race_id','driverId':'driver_id','constructorId':'team_id','grid':'grid_position','positionOrder':'finish_position','position':'position_text'})
 q=q.rename(columns={'raceId':'race_id','driverId':'driver_id','position':'qualifying_position'})
 if 'circuits' in paths:
  c=pd.read_csv(paths['circuits']).rename(columns={'circuitId':'circuit_id','name':'circuit_name'})
  races=races.merge(c[[x for x in ('circuit_id','circuitRef','circuit_name') if x in c]].drop_duplicates('circuit_id'),on='circuit_id',how='left')
 qcols=[c for c in ('q1','q2','q3') if c in q]
 def qseconds(row):
  for x in row.iloc[::-1]:
   if pd.isna(x): continue
   try:
    p=str(x).split(':'); return float(p[-1])+60*float(p[-2]) if len(p)>1 else float(p[0])
   except Exception: return np.nan
  return np.nan
 q['qualifying_time_seconds']=q[qcols].apply(qseconds,axis=1) if qcols else np.nan
 d=results.merge(q[[c for c in ('race_id','driver_id','qualifying_position','qualifying_time_seconds') if c in q]],on=['race_id','driver_id'],how='left')
 d=d.merge(races[[c for c in ('race_id','year','date','event_name','circuitRef','circuit_name') if c in races]],on='race_id',how='left',suffixes=('','_race'))
 if 'drivers' in paths:
  dr=pd.read_csv(paths['drivers']).rename(columns={'driverId':'driver_id'}); d=d.merge(dr[[c for c in ('driver_id','forename','surname','code') if c in dr]],on='driver_id',how='left')
  if {'forename','surname'}<=set(d): d['driver']=(d.forename.fillna('')+' '+d.surname.fillna('')).str.strip()
 if 'constructors' in paths:
  co=pd.read_csv(paths['constructors']).rename(columns={'constructorId':'team_id','name':'team'})
  if {'team_id','team'}<=set(co): d=d.merge(co[['team_id','team']],on='team_id',how='left')
 d['race_id']=d.race_id.astype(str); d['driver']=d.get('driver',d.get('driver_id','')).astype(str); d['team']=d.get('team',d.get('team_id','')).astype(str)
 d['circuit_name']=d.get('circuit_name',d.get('circuitRef',d.get('event_name','')))
 for c,v in {'status':'','points':0,'track_temp_c':np.nan,'straight_line_speed_kph':np.nan,'quali_lap1_delta_seconds':np.nan,'race_pace_seconds':np.nan}.items():
  if c not in d: d[c]=v
 return d

def load_fastf1(seasons=(2023,2024,2025,2026),cache_dir='work/fastf1_cache',telemetry=True):
 """Load race/qualifying sessions from FastF1 (optional alternate loader)."""
 try: import fastf1
 except ImportError as e: raise ImportError('Install fastf1 to use this loader') from e
 Path(cache_dir).mkdir(parents=True,exist_ok=True); fastf1.Cache.enable_cache(str(Path(cache_dir).resolve())); rows=[]
 for y in seasons:
  for _,ev in fastf1.get_event_schedule(int(y),include_testing=False).iterrows():
   if int(ev.get('RoundNumber',0))<1: continue
   try:
    q=fastf1.get_session(int(y),int(ev.RoundNumber),'Q'); r=fastf1.get_session(int(y),int(ev.RoundNumber),'R')
    q.load(telemetry=telemetry,weather=False,laps=True,messages=False); r.load(telemetry=False,weather=True,laps=True,messages=False)
   except Exception: continue
   if q.results is None or r.results is None: continue
   ql=q.laps.copy(); rl=r.laps.copy(); w=getattr(r,'weather_data',pd.DataFrame()); temps=pd.to_numeric(w.get('TrackTemp',pd.Series(dtype=float)),errors='coerce'); temp=float(temps.median()) if temps.notna().any() else np.nan
   lookup={str(z.get('Abbreviation','')):z for _,z in q.results.iterrows()}
   for _,z in r.results.iterrows():
    code=str(z.get('Abbreviation','')); qr=lookup.get(code,pd.Series(dtype=object)); qd=ql[ql.DriverNumber.astype(str).eq(str(qr.get('DriverNumber','')))]; rd=rl[rl.DriverNumber.astype(str).eq(str(z.get('DriverNumber','')))]
    qt=qd.LapTime.dropna().min().total_seconds() if len(qd) and qd.LapTime.notna().any() else np.nan; speed=np.nan
    if telemetry and len(qd):
     try: speed=float(qd.pick_fastest().get_telemetry()['Speed'].max())
     except Exception: pass
    l1=rd[rd.LapNumber.eq(1)].LapTime.dropna(); delta=l1.iloc[0].total_seconds()-qt if len(l1) and np.isfinite(qt) else np.nan
    clean=rd[rd.LapTime.notna()].copy()
    for c in ('PitInTime','PitOutTime'):
     if c in clean: clean=clean[clean[c].isna()]
    rows.append({'race_id':f'{y}-{int(ev.RoundNumber):02d}','year':y,'date':ev.EventDate,'event_name':ev.EventName,'circuit_name':ev.Location,'driver':code,'team':z.get('TeamName',''),'grid_position':pd.to_numeric(z.get('GridPosition'),errors='coerce'),'finish_position':pd.to_numeric(z.get('Position'),errors='coerce'),'status':z.get('Status',''),'points':z.get('Points',0),'qualifying_time_seconds':qt,'straight_line_speed_kph':speed,'track_temp_c':temp,'quali_lap1_delta_seconds':delta,'race_pace_seconds':clean.LapTime.dt.total_seconds().median() if len(clean) else np.nan})
 return pd.DataFrame(rows)

def engineer_features(raw:pd.DataFrame)->pd.DataFrame:
 d=raw.copy()
 for c,v in {'driver':'','team':'','status':'','points':0,'grid_position':np.nan,'qualifying_time_seconds':np.nan,'track_temp_c':np.nan,'straight_line_speed_kph':np.nan,'quali_lap1_delta_seconds':np.nan,'race_pace_seconds':np.nan}.items():
  if c not in d: d[c]=v
 d['date']=pd.to_datetime(d.date if 'date' in d else d.get('year',pd.Series(index=d.index)),errors='coerce');
 if 'year' not in d: d['year']=d.date.dt.year
 d['race_id']=d.race_id.astype(str); ev=d.get('circuit_name',d.get('event_name',pd.Series('',index=d.index))).fillna('').map(_slug)
 vegas25=pd.to_numeric(d.year,errors='coerce').eq(2025)&ev.str.contains('vegas',na=False)
 d['finish_position_model']=pd.to_numeric(d.get('finish_position',np.nan),errors='coerce')
 dsq=vegas25&(d.status.astype(str).str.contains('DSQ|disqual',case=False,na=False)|d.get('position_text',pd.Series('',index=d.index)).astype(str).str.fullmatch('D|DSQ',case=False,na=False))
 pace=pd.to_numeric(d.get('pre_dsq_pace_rank',pd.Series(np.nan,index=d.index)),errors='coerce')
 if 'race_pace_seconds' in d: pace=pace.combine_first(d.loc[vegas25].groupby('race_id').race_pace_seconds.rank(method='average').reindex(d.index))
 d.loc[dsq&pace.notna(),'finish_position_model']=pace[dsq&pace.notna()]; d['target_eligible']=d.finish_position_model.notna(); d.loc[dsq&pace.isna(),'target_eligible']=False
 for c in ('grid_position','qualifying_time_seconds','track_temp_c','straight_line_speed_kph','points','quali_lap1_delta_seconds'): d[c]=pd.to_numeric(d.get(c,np.nan),errors='coerce')
 d['points']=d.points.fillna(0); d['position_gained']=d.grid_position-d.finish_position_model
 d['is_low_drag_cluster']=ev.map(lambda x:int(any(n in x for n in LOW_DRAG)))
 d['straight_line_speed_rank']=d.groupby('race_id').straight_line_speed_kph.rank(ascending=False,method='average')
 d=d.sort_values(['date','race_id'],kind='stable').reset_index(drop=True)
 d['rolling_5_race_points']=d.groupby('driver',sort=False).points.transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean())
 d['rolling_5_race_pace_delta']=d.groupby('driver',sort=False).position_gained.transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean())
 d['positions_gained_avg']=d.rolling_5_race_pace_delta; d['driver_experience_races']=d.groupby('driver',sort=False).cumcount()
 teamdelta=d.groupby('team',sort=False).quali_lap1_delta_seconds.transform(lambda s:s.shift(1).rolling(5,min_periods=1).mean())
 d['cold_track_warmup_index']=d.track_temp_c*teamdelta; d['speed_overtake_strength']=21-d.straight_line_speed_rank+d.positions_gained_avg
 latest=pd.to_numeric(d.year,errors='coerce').max(); d['sample_weight']=.97**(latest-pd.to_numeric(d.year,errors='coerce')).clip(lower=0).fillna(0)
 d.loc[pd.to_numeric(d.year,errors='coerce').eq(2026),'sample_weight']*=3
 return d

def _model(name):
 from sklearn.ensemble import ExtraTreesClassifier
 from sklearn.impute import SimpleImputer
 from sklearn.pipeline import make_pipeline
 if name=='LightGBM':
  from lightgbm import LGBMClassifier
  est=LGBMClassifier(n_estimators=300,learning_rate=.035,num_leaves=15,reg_lambda=2,verbosity=-1,random_state=42)
 elif name=='XGBoost':
  from xgboost import XGBClassifier
  est=XGBClassifier(n_estimators=300,learning_rate=.035,max_depth=3,min_child_weight=4,reg_lambda=2,eval_metric='logloss',n_jobs=4,random_state=42)
 elif name=='ExtraTrees': est=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,max_features=.8,n_jobs=-1,random_state=42)
 else: raise ValueError(name)
 return make_pipeline(SimpleImputer(strategy='median',add_indicator=True),est)
def _fit(m,x,y,w): m.fit(x,y,**{f'{m.steps[-1][0]}__sample_weight':np.asarray(w)}); return m
def _softmax(s):
 z=np.asarray(s,float); z-=z.max(); p=np.exp(np.clip(z,-40,40)); return p/p.sum()
def _fixed_sum(s,n):
 s=np.asarray(s,float); s=np.nan_to_num(s,nan=float(np.nanmedian(s)) if np.isfinite(s).any() else 0.); lo,hi=-50.,50.
 for _ in range(80):
  m=(lo+hi)/2; p=1/(1+np.exp(-np.clip(s+m,-60,60)))
  if p.sum()<n: lo=m
  else: hi=m
 return 1/(1+np.exp(-np.clip(s+(lo+hi)/2,-60,60)))

def compare_models(data,n_splits=5):
 from sklearn.model_selection import GroupKFold
 d=data[data.target_eligible&data.finish_position_model.notna()].copy().reset_index(drop=True); groups=d.race_id.astype(str).to_numpy()
 if len(np.unique(groups))<2: raise ValueError('At least two races are required for GroupKFold.')
 yw=d.finish_position_model.eq(1).astype(int).to_numpy(); yp=d.finish_position_model.le(3).astype(int).to_numpy(); folds=list(GroupKFold(min(n_splits,len(np.unique(groups)))).split(d[FEATURES],yw,groups)); rows=[]
 for name in ALGORITHMS:
  losses=[]; top1=[]; recall=[]; exact=[]; briers=[]
  for fold,(tr,va) in enumerate(folds,1):
   print(f'CV {name} fold {fold}/{len(folds)}',flush=True); v=d.iloc[va].reset_index(drop=True)
   mw=_fit(_model(name),d.iloc[tr][FEATURES],yw[tr],d.iloc[tr].sample_weight); mp=_fit(_model(name),d.iloc[tr][FEATURES],yp[tr],d.iloc[tr].sample_weight)
   pw=mw.predict_proba(v[FEATURES])[:,1]; pp=mp.predict_proba(v[FEATURES])[:,1]
   for _,ix in v.groupby('race_id',sort=False).groups.items():
    ids=np.asarray(list(ix),int); prob=_softmax(np.log(np.clip(pw[ids],1e-8,1))); actual=np.flatnonzero(yw[va][ids])
    if len(actual): losses.append(-math.log(max(float(prob[actual[0]]),1e-12))); top1.append(int(np.argmax(prob)==actual[0]))
    logit=np.log(np.clip(pp[ids],1e-8,1-1e-8)/np.clip(1-pp[ids],1e-8,1)); p3=_fixed_sum(logit,min(3,len(ids))); y=yp[va][ids]; best=np.argsort(p3)[-min(3,len(ids)):]; hit=int(y[best].sum()); den=max(1,min(3,int(y.sum())))
    recall.append(hit/den); exact.append(int(hit==den)); briers.extend(((p3-y)**2).tolist())
  rows.append({'algorithm':name,'mean_winner_log_loss':np.mean(losses),'winner_top1_accuracy':np.mean(top1),'podium_top3_recall':np.mean(recall),'exact_podium_accuracy':np.mean(exact),'podium_brier_score':np.mean(briers),'folds':len(folds)})
 scores=pd.DataFrame(rows).sort_values('mean_winner_log_loss').reset_index(drop=True)
 return scores,str(scores.iloc[0].algorithm)

def fit_best_model(data,algorithm):
 d=data[data.target_eligible&data.finish_position_model.notna()].copy(); models={}
 for name,y in [('win',d.finish_position_model.eq(1).astype(int)),('podium',d.finish_position_model.le(3).astype(int))]: models[name]=_fit(_model(algorithm),d[FEATURES],y,d.sample_weight)
 return models,{'algorithm':algorithm,'features':FEATURES,'training_rows':len(d),'history_races':int(d.race_id.nunique())}

def make_bundle(models,metadata,history): return {'artifact_type':'general-f1-model-v1','models':models,'metadata':metadata,'history':history.copy()}

def list_race_events(years=None):
 """List schedule events and whether OpenF1 has a completed qualifying session."""
 now_dt=datetime.now(timezone.utc); now=now_dt.date(); years=list(years or [now.year,now.year+1]); events=[]
 for year in years:
  try: races=_api(f'https://api.jolpi.ca/ergast/f1/{year}.json?limit=100')['MRData']['RaceTable']['Races']
  except Exception: continue
  try: sessions=_api(f'https://api.openf1.org/v1/sessions?year={year}&session_name=Qualifying')
  except Exception: sessions=[]
  for race in races:
   if pd.Timestamp(race['date']).date()<now: continue
   circuit=race.get('Circuit',{}); loc=circuit.get('Location',{}).get('locality',''); name=race.get('raceName','')
   match=next((s for s in sessions if not s.get('is_cancelled',False) and (_slug(s.get('meeting_name',''))==_slug(name) or _slug(s.get('location',''))==_slug(loc) or _slug(s.get('circuit_short_name',''))==_slug(loc))),None)
   q_end=pd.to_datetime(match.get('date_end'),utc=True,errors='coerce') if match else pd.NaT
   completed=bool(match is not None and pd.notna(q_end) and q_end.to_pydatetime()<=now_dt)
   events.append({'race_id':f"{year}-{int(race['round']):02d}",'year':year,'round':int(race['round']),'date':race['date'],'event_name':name,'circuit_name':circuit.get('circuitName',loc),'location':loc,'session_key':match.get('session_key') if match else None,'qualifying_scheduled':match is not None,'qualifying_available':completed})
 return sorted(events,key=lambda x:(x['date'],x['round']))

def fetch_qualifying_entries(event):
 """Fetch completed qualifying laps, teams, speed trap, and Q-session track temp."""
 if not event.get('session_key'): raise RuntimeError('Qualifying has not been published for this race yet.')
 key=event['session_key']; laps=_api(f'https://api.openf1.org/v1/laps?session_key={key}'); drivers=_api(f'https://api.openf1.org/v1/drivers?session_key={key}')
 dmap={str(x['driver_number']):x for x in drivers}; entries={}
 for lap in laps:
  num=str(lap.get('driver_number')); dr=dmap.get(num,{}); code=dr.get('name_acronym'); sec=lap.get('lap_duration')
  if not code or sec is None or lap.get('is_pit_out_lap',False): continue
  row=entries.setdefault(code,{'driver':code,'team':dr.get('team_name',''),'best':float(sec),'straight_line_speed_kph':np.nan})
  row['best']=min(row['best'],float(sec))
  if lap.get('st_speed') is not None: row['straight_line_speed_kph']=max(float(row['straight_line_speed_kph']) if np.isfinite(row['straight_line_speed_kph']) else 0.,float(lap['st_speed']))
 e=pd.DataFrame(entries.values())
 if e.empty: raise RuntimeError('Qualifying exists but timing laps are not available yet.')
 e['qualifying_time_seconds']=e.best; e['grid_position']=e.best.rank(method='first').astype(int)
 e['race_id']=event['race_id']; e['year']=event['year']; e['date']=event['date']; e['event_name']=event['event_name']; e['circuit_name']=event['circuit_name']
 try:
  weather=_api(f'https://api.openf1.org/v1/weather?session_key={key}'); temp=[float(x['track_temperature']) for x in weather if x.get('track_temperature') is not None]
  e['track_temp_c']=float(np.median(temp)) if temp else np.nan
 except Exception: e['track_temp_c']=np.nan
 return e.drop(columns='best')

def predict_with_bundle(bundle,entries):
 e=entries.copy(); req={'race_id','date','driver','team','grid_position','qualifying_time_seconds'}
 if req-set(e): raise ValueError('Entries missing: '+', '.join(sorted(req-set(e))))
 if e.race_id.nunique()!=1: raise ValueError('Predict one race at a time.')
 n=max(3,int(np.ceil(.8*len(e))))
 for c in ('grid_position','qualifying_time_seconds'):
  if pd.to_numeric(e[c],errors='coerce').notna().sum()<n: raise ValueError(f'Need {n} valid {c} values for a post-qualifying prediction.')
 rid=str(e.race_id.iloc[0]); e['points']=0; e['finish_position']=np.nan; e['status']=''
 h=bundle['history'].copy(); h=h[h.race_id.astype(str).ne(rid)]; cutoff=pd.to_datetime(e.date,errors='coerce').min(); h=h[pd.to_datetime(h.date,errors='coerce')<cutoff]
 all_rows=engineer_features(pd.concat([h,e],ignore_index=True,sort=False)); target=all_rows[all_rows.race_id.astype(str).eq(rid)].copy()
 wp=bundle['models']['win'].predict_proba(target[FEATURES])[:,1]; pp=bundle['models']['podium'].predict_proba(target[FEATURES])[:,1]
 wl=np.log(np.clip(wp,1e-8,1-1e-8)/np.clip(1-wp,1e-8,1)); pl=np.log(np.clip(pp,1e-8,1-1e-8)/np.clip(1-pp,1e-8,1))
 target['win_probability']=_softmax(wl); target['podium_probability']=_fixed_sum(pl,min(3,len(target)))
 target=target.sort_values(['podium_probability','win_probability'],ascending=False).reset_index(drop=True)
 target['predicted_podium_position']=pd.array([i+1 if i<3 else None for i in range(len(target))],dtype='Int64')
 return target
