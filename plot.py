import numpy as np
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Div,
    CustomJS,
    LinearColorMapper,
    ColorBar,
)
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256, Turbo256, TolRainbow19
from bokeh.layouts import column, row



def create_hover_plot(umap_projection: pd.DataFrame, chunks_df: pd.DataFrame):
    x = umap_projection.iloc[:, 0].values
    y = umap_projection.iloc[:, 1].values
    texts = chunks_df["text"].values
    values = np.arange(len(x))
    sizes = chunks_df["Length"].values
    
    # Create a colormap
    mapper = linear_cmap(
        field_name="values",
        palette=Turbo256,
        low=np.min(values),
        high=np.max(values),
    )

    p = figure(
        width=1200,
        height=1000,
        tools="box_zoom,wheel_zoom,pan,reset",
        title="Hover over points",
    )
    output_file("hover_callback.html")
    sizes = ((sizes - np.min(sizes) + 1 ) / (np.max(sizes) - np.min(sizes)) + 0.01 ) * 0.1
    source = ColumnDataSource(
        data=dict(x=x, y=y, text=texts, values=values, sizes=sizes)
    )
    cr = p.circle(
        x="x",
        y="y",
        color=mapper,
        radius="sizes",
        alpha=0.9,
        hover_alpha=1.0,
        line_color="white",  # Add white marker border
        line_width=1,  # Set the width of the marker border
        source=source,
    )

    # Add a color bar to the plot
    color_bar = ColorBar(color_mapper=mapper["transform"], width=8, location=(0, 0))
    p.add_layout(color_bar, "right")

    div = Div(
        width=1000, height=100, height_policy="fixed", styles={"font-size": "16pt"}
    )

    # add a hover tool that sets the text for a hovered circle
    code = """
    const data = {'text': []}
    const indices = cb_data.index.indices
    for (let i = 0; i < indices.length; i++) {
        const index = indices[i]
        data['text'].push(source.data['text'][index])
    }
    div.text = data['text'].join('\\n')
    """

    callback = CustomJS(args={"source": source, "div": div}, code=code)

    tooltips = [
        ("Value", "@values"),
    ]
    p.add_tools(HoverTool(tooltips=tooltips, callback=callback, renderers=[cr]))

    layout = row(p, div)

    show(layout)
