"""Small inference-only UI for predictions across upcoming F1 race weekends."""
from datetime import datetime,timezone
from pathlib import Path
import pickle
import pandas as pd
import streamlit as st
import f1_race_pipeline as f1

st.set_page_config(page_title='F1 Race Predictor',page_icon='🏁',layout='wide')
st.title('🏁 Formula 1 Race Predictor')
st.caption('One pre-trained model for the full calendar. Choose an upcoming Grand Prix; qualifying data is fetched automatically when it is published.')
model_path=Path(__file__).with_name('f1_all_races_model.pkl')
if not model_path.exists():
    st.error('Pre-trained model artifact is missing. Run `python outputs/train_model.py` offline once; Streamlit does not train.')
    st.stop()

@st.cache_resource
def load_model(path):
    with Path(path).open('rb') as fp:
        return pickle.load(fp)

bundle=load_model(str(model_path)); meta=bundle['metadata']
metrics=meta.get('cv_scores',[])
cols=st.columns(3)
cols[0].metric('Selected model',meta['algorithm'])
cols[1].metric('Training races',str(meta.get('history_races',bundle['history'].race_id.nunique())))
cols[2].metric('Winner CV log loss',f"{metrics[0]['mean_winner_log_loss']:.3f}" if metrics else '—')
with st.expander('Cross-validation model comparison'):
    if metrics: st.dataframe(pd.DataFrame(metrics),use_container_width=True,hide_index=True)

now=datetime.now(timezone.utc).year
if st.button('Refresh upcoming race list'):
    st.cache_data.clear()

@st.cache_data(ttl=900)
def get_events(years):
    return f1.list_race_events(years)

events=get_events(tuple(range(now,now+2)))
if not events:
    st.info('No upcoming races were returned by the public calendar API.')
    st.stop()

def label(e):
    status='qualifying available' if e['qualifying_available'] else 'qualifying pending'
    return f"{e['date']} · {e['event_name']} · {status}"

selected=st.selectbox('Upcoming Grand Prix',events,format_func=label)
if selected['qualifying_available']:
    st.success(f"Qualifying is available for {selected['event_name']}. The app will use its best lap times and speed traps.")
else:
    st.info('Prediction opens after this event’s qualifying session is published.')

if st.button('Fetch qualifying and predict podium',type='primary',disabled=not selected['qualifying_available']):
    try:
        with st.spinner('Fetching post-qualifying laps, grid order, speed traps, and session weather…'):
            entries=f1.fetch_qualifying_entries(selected)
            pred=f1.predict_with_bundle(bundle,entries)
        st.session_state['race_entries']=entries
        st.session_state['race_prediction']=pred
    except Exception as exc:
        st.error(str(exc))

entries=st.session_state.get('race_entries'); pred=st.session_state.get('race_prediction')
if pred is not None and entries is not None:
    if selected['race_id']!=str(pred.race_id.iloc[0]):
        st.caption('The visible prediction is for the previously selected event. Click the prediction button after selecting another race.')
    else:
        temp=entries.track_temp_c.dropna().iloc[0] if entries.track_temp_c.notna().any() else None
        st.subheader(f"Predicted podium · {selected['event_name']}")
        st.caption(f"Track temperature feature: {temp:.1f} °C" if temp is not None else 'Track temperature was unavailable from qualifying; the model imputes it.')
        top=pred.dropna(subset=['predicted_podium_position']).sort_values('predicted_podium_position')
        cards=st.columns(3)
        for card,(_,row) in zip(cards,top.iterrows()):
            card.metric(f"P{int(row.predicted_podium_position)} · {row.driver}",f"{row.podium_probability:.1%} podium chance",f"{row.win_probability:.1%} win chance")
            card.caption(row.team)
        st.subheader('Full field')
        show=pred[[c for c in ('predicted_podium_position','driver','team','grid_position','qualifying_time_seconds','straight_line_speed_rank','positions_gained_avg','win_probability','podium_probability') if c in pred]].copy()
        st.dataframe(show,use_container_width=True,hide_index=True)
        st.caption(f"Win probability total: {pred.win_probability.sum():.1%} · podium marginals: {pred.podium_probability.sum():.2f} expected places")
        st.download_button('Download race prediction CSV',pred.to_csv(index=False).encode(),f"{selected['race_id']}_prediction.csv",'text/csv')
        st.info('Grid order is initialized from qualifying lap position. If the official starting grid changes due to penalties, refresh the qualifying entry data before race start.')

st.divider()
st.caption('Training happens offline in train_model.py. The same fitted model supports every circuit; only the race selection and its completed qualifying session change.')
