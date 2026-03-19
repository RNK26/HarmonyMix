"""
audio_radar.py — Streamlit component: renders a radar/spider chart of audio features.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


AUDIO_FEATURES = [
    'danceability', 'energy', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
]


def render_audio_radar(row: pd.Series, title: str = "Audio Feature Profile") -> None:
    """Renders a Plotly radar chart for the audio features of a single track.

    Args:
        row:   A pandas Series representing one row of the tracks DataFrame.
        title: Chart title shown above the radar.
    """
    values = [float(row.get(f, 0.0)) for f in AUDIO_FEATURES]
    # Close the polygon by repeating the first value
    labels = AUDIO_FEATURES + [AUDIO_FEATURES[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values_closed,
            theta=labels,
            fill='toself',
            line_color='#1DB954',
            fillcolor='rgba(29, 185, 84, 0.25)',
            name=str(row.get('name', '')).title(),
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont_size=9),
            angularaxis=dict(tickfont_size=11),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False,
        title=dict(text=title, font=dict(color='#1DB954', size=14)),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
    )
    st.plotly_chart(fig, width="stretch")
