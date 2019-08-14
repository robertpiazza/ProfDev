# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:06:33 2019

#point conda or python command line to the directory, and run 
#bokeh serve --show gapminder_script.py
#ctrol-c to interrupt server


@author: 593787
"""
#Imports 

import pandas as pd
from bokeh.io import curdoc, output_file 
from bokeh.layouts import row  
from bokeh.models import ColumnDataSource, Select, HoverTool,  CategoricalColorMapper, Slider, RadioGroup
from bokeh.plotting import figure 
from bokeh.layouts import widgetbox
from bokeh.palettes import Spectral6

data = pd.read_csv('data/gapminder_tidy.csv', index_col = 1)

# Make the ColumnDataSource: source
source = ColumnDataSource(data={
    'x'       : data.loc[1970].fertility,
    'y'       : data.loc[1970].life,
    'country'      : data.loc[1970].Country,
    's'      : (data.loc[1970].population / 10000000) + 2,
    'size'   : (data.loc[1970].population),
    'region'      : data.loc[1970].region,
})

# Save the minimum and maximum values of the fertility column: xmin, xmax
xmin, xmax = min(data.fertility), max(data.fertility)

# Save the minimum and maximum values of the life expectancy column: ymin, ymax
ymin, ymax = min(data.life), max(data.life)

# Create the figure: plot
plot = figure(title='Gapminder Data for 1970', plot_height=400, plot_width=700,
              x_range=(xmin, xmax), y_range=(ymin, ymax), tools='box_select, lasso_select, pan, box_zoom, wheel_zoom, reset, help')


# Set the x-axis label
plot.xaxis.axis_label ='Fertility (children per woman)'

# Set the y-axis label
plot.yaxis.axis_label = 'Life Expectancy (years)'

axis_dict = {'life':'Life Expectancy (years)',
             'fertility': 'Fertility (children per woman)', 
             'child_mortality': 'Child Mortality (0-5 year-olds dying per 1000 born)',
             'gdp': 'GDP (Gross Domestic Product)',
             'pop': 'Population'}

# Make a list of the unique values from the region column: regions_list
regions_list = data.region.unique().tolist()

# Make a color mapper: color_mapper
color_mapper = CategoricalColorMapper(factors=regions_list, palette=Spectral6)

# Add the color mapper to the circle glyph
plot.circle(x='x', y='y', size = 's', fill_alpha=0.8, source=source,
            color=dict(field='region', transform =color_mapper), legend='region')

# Set the legend.location attribute of the plot to 'top_right' (default)
plot.legend.location = 'top_right'

# Add the plot to the current document and add the title
curdoc().title = 'Gapminder'


# Create a HoverTool: hover
hover = HoverTool(tooltips = [('Country', '@country'),
                              ('Population', '@size')])

# Add the HoverTool to the plot
plot.add_tools(hover)


# Define the callback: update_plot
def update_plot(attr, old, new):  

    # Read the current value off the slider and 2 dropdowns: yr, x, y
    yr = slider.value
    x = x_select.value
    y = y_select.value
    r = r_select.value
    region_choose = (data.index == yr) & (data.region == r)

    if r =='all':
        s_data = (data.loc[yr]['population']/10000000)+2
        size_data = data.loc[yr]['population']
    else:
        s_data = (data.loc[region_choose]['population']/10000000)+2
        size_data = data.loc[region_choose]['population']
        
    # Label axes of plot
    plot.xaxis.axis_label = axis_dict[x]
    plot.yaxis.axis_label = axis_dict[y]
    # Set new_data
    if r == 'all':
        new_data = {
            'x'       : data.loc[yr][x],
            'y'       : data.loc[yr][y],
            'country' : data.loc[yr].Country,
            's'       : s_data,
            'size'   : size_data,
            'region'  : data.loc[yr].region,
        }
    else:
        new_data = {
            'x'       : data.loc[region_choose][x],
            'y'       : data.loc[region_choose][y],
            'country' : data.loc[region_choose].Country,
            's'       : s_data,
            'size'   : size_data,
            'region'  : data.loc[region_choose].region,
        }                
    # Assign new_data to source.data
    source.data = new_data
    
    # Set the range of all axes
    plot.x_range.start = min(data[x])
    plot.x_range.end = max(data[x])
    plot.y_range.start = min(data[y])
    plot.y_range.end = max(data[y])

    # Add title to plot
    plot.title.text = 'Gapminder data for %d' % yr

def update_legend(attr, old, new):
    legend_locs = ["top_left","top_right", "bottom_left", "bottom_right"]
    l = l_select.active
    plot.legend.location = legend_locs[l]
    

    

# Create a dropdown slider widget: slider
slider = Slider(start=1970, end=2010, step=1, value=1970, title='Year')
# Attach the callback to the 'value' property of slider
slider.on_change('value', update_plot)


# Create a dropdown Select widget for the x data: x_select
x_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='fertility',
    title='x-axis data'
)

# Attach the update_plot callback to the 'value' property of x_select
x_select.on_change('value', update_plot)

# Create a dropdown Select widget for the y data: y_select
y_select = Select(
    options=['fertility', 'life', 'child_mortality', 'gdp'],
    value='life',
    title='y-axis data'
)

# Attach the update_plot callback to the 'value' property of y_select
y_select.on_change('value', update_plot)

legend_locs = ["top_left","top_right", "bottom_left", "bottom_right"]

# Create a radioGroup widget for the legend : l_select
l_select = RadioGroup(labels=legend_locs, 
                      active = 1,
                      name = 'Legend Location')

# Attach the update_plot callback to the 'value' property of r_select (region)
l_select.on_change('active', update_legend)

# Create a dropdown Select widget for the y data: y_select
r_select = Select(
    options=['all']+regions_list,
    value='all',
    title='Region'
)

# Attach the update_plot callback to the 'value' property of y_select
r_select.on_change('value', update_plot)

    

# Create layout and add to current document
layout = row(widgetbox(slider, x_select, y_select, r_select, l_select), plot)
output_file('gapminder.html')
curdoc().add_root(layout)

