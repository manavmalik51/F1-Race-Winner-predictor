from __future__ import annotations
from pathlib import Path
from datetime import date
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import hashlib, json, time, re
import numpy as np
import pandas as pd

CACHE=Path('work/public_api_cache'); CACHE.mkdir(parents=True,exist_ok=True)
TODAY=pd.Timestamp('2026-11-14')

def get_json(url):
    f=CACHE/(hashlib.sha1(url.encode()).hexdigest()+'.json')
    if f.exists(): return json.loads(f.read_text())
    req=Request(url,headers={'User-Agent':'F1-Vegas-ML/1.0 contact: public-data-research'})
    for attempt in range(5):
        try:
            with urlopen(req,timeout=40) as r: data=json.loads(r.read())
            f.write_text(json.dumps(data)); time.sleep(.85); return data
        except HTTPError as e:
            if e.code==429: time.sleep(15*(attempt+1))
            elif attempt==4: raise
            else: time.sleep(2*(attempt+1))
        except Exception:
            if attempt==4: raise
            time.sleep(2*(attempt+1))

def paged(endpoint):
    offset=0; out=[]
    while True:
        d=get_json(f'https://api.jolpi.ca/ergast/f1/{endpoint}?limit=100&offset={offset}')['MRData']
        races=d.get('RaceTable',{}).get('Races',[]); out.extend(races)
        total=int(d.get('total',0)); limit=int(d.get('limit',100))
        offset+=limit
        if not races or offset>=total: break
    return out

def lap_seconds(value):
    if not value: return np.nan
    parts=str(value).split(':')
    try: return float(parts[-1])+60*float(parts[-2]) if len(parts)>1 else float(parts[0])
    except ValueError: return np.nan

race_meta={}; results=[]; quali=[]
for year in range(2023,2027):
    schedule=paged(f'{year}')
    schedule=[r for r in schedule if pd.Timestamp(r['date'])<=TODAY]
    for r in schedule:
        rid=f"{year}-{int(r['round']):02d}"
        race_meta[rid]={'race_id':rid,'year':year,'round':int(r['round']),'date':r['date'],
            'event_name':r['raceName'],'circuit_name':r['Circuit']['circuitName'],
            'location':r['Circuit']['Location']['locality'],'api_race_id':None}
    print('Schedule',year,len(schedule),flush=True)
    for r in paged(f'{year}/results'):
        rid=f"{year}-{int(r['round']):02d}"; race_meta.setdefault(rid,{'race_id':rid,'year':year,'round':int(r['round']),'date':r['date'],'event_name':r['raceName'],'circuit_name':r['Circuit']['circuitName'],'location':r['Circuit']['Location']['locality']})
        for x in r.get('Results',[]):
            d=x['Driver']; c=x.get('Constructor',{})
            status=x.get('status',''); pos=x.get('position')
            if 'disqual' in str(status).lower(): pos=None
            results.append({'race_id':rid,'year':year,'date':r['date'],'event_name':r['raceName'],'circuit_name':r['Circuit']['circuitName'],'driver_id':d['driverId'],'driver':d.get('code') or (d['givenName']+' '+d['familyName']),'team':c.get('name',''),'grid_position':pd.to_numeric(x.get('grid'),errors='coerce'),'finish_position':pd.to_numeric(pos,errors='coerce'),'position_text':x.get('positionText',''),'status':status,'points':pd.to_numeric(x.get('points'),errors='coerce'),'fastest_lap_speed':pd.to_numeric(x.get('FastestLap',{}).get('AverageSpeed',{}).get('speed'),errors='coerce')})
    for r in paged(f'{year}/qualifying'):
        rid=f"{year}-{int(r['round']):02d}"
        for x in r.get('QualifyingResults',[]):
            d=x['Driver']; qtime=next((x[k] for k in ('Q3','Q2','Q1') if x.get(k)),None)
            quali.append({'race_id':rid,'driver_id':d['driverId'],'qualifying_position':pd.to_numeric(x.get('position'),errors='coerce'),'qualifying_time_seconds':lap_seconds(qtime)})

df=pd.DataFrame(results).merge(pd.DataFrame(quali),on=['race_id','driver_id'],how='left')
df['track_temp_c']=np.nan; df['straight_line_speed_kph']=np.nan; df['quali_lap1_delta_seconds']=np.nan; df['race_pace_seconds']=np.nan

# Lap one timing for every completed race: feeds only shifted historical warm-up features.
lap1={}
for rid,row in race_meta.items():
    if rid not in set(df.race_id): continue
    yr,rd=rid.split('-')
    try:
        j=get_json(f'https://api.jolpi.ca/ergast/f1/{yr}/{int(rd)}/laps/1.json?limit=100')['MRData']['RaceTable']['Races']
        for lap in (j[0].get('Laps',[]) if j else []):
            for t in lap.get('Timings',[]): lap1[(rid,t['driverId'])]=lap_seconds(t.get('time'))
    except Exception as exc: print('lap1 skip',rid,repr(exc),flush=True)
df['lap1_seconds']=[lap1.get((r.race_id,r.driver_id),np.nan) for r in df.itertuples()]
df['quali_lap1_delta_seconds']=df.lap1_seconds-df.qualifying_time_seconds

# Public OpenF1 timing supplies Vegas race-lap pace for the post-race DSQ correction.
df['pre_dsq_pace_rank']=np.nan

# Public OpenF1 meeting/session records provide qualifying speed-trap telemetry.
open_sessions=[]
for year in range(2023,2027):
    for kind in ('Race','Qualifying'):
        try: open_sessions.extend(get_json(f'https://api.openf1.org/v1/sessions?year={year}&session_name={kind}'))
        except Exception as exc: print('sessions miss',year,kind,repr(exc),flush=True)

def norm(s): return re.sub(r'[^a-z0-9]+','',str(s).lower())
# Weather table is a compact public FastF1-derived dataset (recent years only).
def github_csv(path):
    obj=get_json(f'https://api.github.com/repos/Arths17/f1-dataset/contents/output/{path}?ref=main')
    if obj.get('encoding')=='base64':
        import base64,io
        return pd.read_csv(io.BytesIO(base64.b64decode(obj['content'])))
    blob=get_json(obj['git_url'])
    import base64,io
    return pd.read_csv(io.BytesIO(base64.b64decode(blob['content'])))
try:
    source_races=github_csv('races.csv'); weather=github_csv('weather.csv')
    source_races['race_id']=source_races.raceId.astype(str)
    source_races['key']=source_races.year.astype(str)+'-'+source_races['round'].astype(int).map(lambda x:f'{x:02d}')
    # Re-read/normalize explicitly because CSV raceId field names differ by source.
    wr=github_csv('weather.csv'); wr=wr.rename(columns={'raceId':'api_race_id','trackTempAvg':'track_temp_c'})
    rr=source_races.rename(columns={'raceId':'api_race_id'})[['api_race_id','key']]
    weather_map=wr.merge(rr,on='api_race_id',how='left').dropna(subset=['key']).set_index('key').track_temp_c.to_dict()
    df['track_temp_c']=df.race_id.map(weather_map)
except Exception as exc:
    print('weather dataset unavailable:',repr(exc),flush=True)

# Backfill Las Vegas track temperature for the small pre-2025 sample from OpenF1.
for rid in ('2023-22','2024-22'):
    meta=race_meta.get(rid)
    if not meta: continue
    sessions=[s for s in open_sessions if s.get('year')==meta['year'] and s.get('session_name')=='Race' and 'vegas' in norm(s.get('location','')+' '+s.get('meeting_name',''))]
    if sessions:
        try:
            wx=get_json(f"https://api.openf1.org/v1/weather?session_key={sessions[0]['session_key']}")
            t=[float(x['track_temperature']) for x in wx if x.get('track_temperature') is not None]
            if t: df.loc[df.race_id.eq(rid),'track_temp_c']=float(np.median(t))
        except Exception as exc: print('Vegas weather miss',rid,repr(exc),flush=True)

low=('monza','baku','jeddah','lasvegas')
for sess in open_sessions:
    if sess.get('session_name')!='Qualifying' or not any(x in norm(sess.get('circuit_short_name','')) for x in low): continue
    y=sess['year']; name=norm(sess.get('meeting_name','')); loc=norm(sess.get('location',''))
    matches=[rid for rid,m in race_meta.items() if m['year']==y and (norm(m['event_name'])==name or norm(m['location'])==loc)]
    if not matches: continue
    rid=matches[0]
    try:
        laps=get_json(f"https://api.openf1.org/v1/laps?session_key={sess['session_key']}")
        drivers=get_json(f"https://api.openf1.org/v1/drivers?session_key={sess['session_key']}")
        code={str(x['driver_number']):x.get('name_acronym') for x in drivers}
        speeds={}
        for lap in laps:
            sp=lap.get('st_speed')
            if sp is not None and not lap.get('is_pit_out_lap',False):
                n=code.get(str(lap.get('driver_number')))
                if n: speeds[n]=max(speeds.get(n,0),float(sp))
        df.loc[df.race_id.eq(rid),'straight_line_speed_kph']=df.loc[df.race_id.eq(rid),'driver'].map(speeds)
        print('speed telemetry',rid,len(speeds),flush=True)
    except Exception as exc: print('speed miss',rid,repr(exc),flush=True)

# Race-lap median rank for 2025 Vegas DSQ labels.
vegas_races=[s for s in open_sessions if s.get('year')==2025 and s.get('session_name')=='Race' and 'vegas' in norm(s.get('location','')+' '+s.get('meeting_name',''))]
if vegas_races:
    sess=vegas_races[0]
    try:
        laps=get_json(f"https://api.openf1.org/v1/laps?session_key={sess['session_key']}")
        drivers=get_json(f"https://api.openf1.org/v1/drivers?session_key={sess['session_key']}")
        code={str(x['driver_number']):x.get('name_acronym') for x in drivers}; paces={}
        for lap in laps:
            dur=lap.get('lap_duration')
            if dur and not lap.get('is_pit_out_lap',False):
                c=code.get(str(lap.get('driver_number')))
                if c: paces.setdefault(c,[]).append(float(dur))
        ranks=pd.Series({k:np.nanmedian(v) for k,v in paces.items()}).rank(method='average').to_dict()
        df.loc[df.race_id.eq('2025-22'),'pre_dsq_pace_rank']=df.loc[df.race_id.eq('2025-22'),'driver'].map(ranks)
        print('Vegas 2025 DSQ pace ranks',len(ranks),flush=True)
    except Exception as exc: print('Vegas pace miss',repr(exc),flush=True)

out=Path('work/f1_history_source.csv'); df.to_csv(out,index=False)
print('saved',out,'rows',len(df),'races',df.race_id.nunique(),'years',df.groupby('year').race_id.nunique().to_dict(),flush=True)
print('weather rows',df.groupby('year').track_temp_c.apply(lambda x:x.notna().sum()).to_dict(),'speed rows',df.groupby('year').straight_line_speed_kph.apply(lambda x:x.notna().sum()).to_dict(),flush=True)
