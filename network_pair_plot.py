import pandas as pd
import holoviews as hv
from bokeh.io import output_file, save

hv.extension("bokeh")

# Load your network pair counts
df = pd.read_csv("network_pair_counts.csv")

# Holoviews expects (source, target, value)
chord_data = [(row.net_a, row.net_b, row.count) for _, row in df.iterrows()]

# Unique network labels
nodes = list(sorted(set(df.net_a).union(df.net_b)))

# Create chord diagram
chord = hv.Chord(chord_data).opts(
    width=800,
    height=800,
    cmap="Category20",
    labels="name",
    node_color="index",
    edge_color="source",
    edge_cmap="Category20",
    title="Network Interaction Chord Diagram"
)

output_file("network_chord.html")
save(hv.render(chord, backend="bokeh"))
print("Saved network_chord.html")
