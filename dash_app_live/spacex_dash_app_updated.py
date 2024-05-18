import pandas as pd
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from folium.plugins import MousePosition
from folium.features import DivIcon

# Read the SpaceX launch data into a pandas dataframe

spacex_df=pd.read_csv("spacex_df_folium.csv")
spacex_df = spacex_df[['Launch Site', 'Lat', 'Long', 'class']]
launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()
launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
launch_sites_df
nasa_coordinate = [29.559684888503615, -95.0830971930759]
spacex_df_plot = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df_plot['Payload Mass (kg)'].max()
min_payload = spacex_df_plot['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)
server = app.server
# Create an app layout
app.layout = html.Div(children=[
    html.H1('SpaceX Launch Records Dashboard',
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    # TASK 1: Add a dropdown list to enable Launch Site selection
    # The default select value is for ALL sites
    dcc.Dropdown(id='site-dropdown',
                 options=[
                     {'label': 'All Sites', 'value': 'ALL'},
                     {'label': 'CCAFS LC-40', 'value': 'CCAFS LC-40'},
                     {'label': 'CCAFS SLC-40', 'value': 'CCAFS SLC-40'},
                     {'label': 'KSC LC-39A', 'value': 'KSC LC-39A'},
                     {'label': 'VAFB SLC-4E', 'value': 'VAFB SLC-4E'}
                 ],
                 value='ALL',
                 placeholder="Select a launch site",
                 searchable=True),
    html.Br(),

    # TASK 2: Add a pie chart to show the total successful launches count for all sites
    # If a specific launch site was selected, show the Success vs. Failed counts for the site
    html.Div(dcc.Graph(id='success-pie-chart'),),
    html.Br(),

    html.P("Payload range (Kg):"),
    # TASK 3: Add a slider to select payload range
    dcc.RangeSlider(id='payload-slider',
                    min=0, max=10000, step=1000,
                    marks={int(min_payload): str(int(min_payload)),
                           int(max_payload): str(int(max_payload))},
                    value=[min_payload, max_payload]),

    # TASK 4: Add a scatter chart to show the correlation between payload and launch success
    html.Div(dcc.Graph(id='success-payload-scatter-chart'),),
    html.Br(),
    html.H2("Launch Sites on Map",style={'textAlign': 'center', 'color': '#503D36', 'font-size': 30}),
    html.Iframe(id='folium-map', width='70%', height=400, style={'padding': '10px','margin': 'auto', 'display': 'block'}),
    html.Div(style={"margin":"180px 20px"})
])
from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    hav_theta = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    theta = 2 * atan2(sqrt(hav_theta), sqrt(1 - hav_theta))

    distance = R * theta
    return distance

def create_folium_map():
    site_map = folium.Map(location=nasa_coordinate, zoom_start=2)
    circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
    marker = folium.map.Marker(
        nasa_coordinate,
        # Create an icon as a text label
        icon=DivIcon(
            icon_size=(20,20),
            icon_anchor=(0,0),
            html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
            )
        )
    site_map.add_child(circle)
    site_map.add_child(marker)
    marker_cluster = MarkerCluster()
    spacex_df["marker_color"] = list(map(lambda x:"#00FF00" if x==1 else "#FF0000",spacex_df["class"]))
    site_map.add_child(marker_cluster)
    def pop_up(x):
        if x == 1:
            return "Success"
        return "Failure"
    for index, record in spacex_df.iterrows():
    
    # TODO: Create and add a Marker cluster to the site map
        marker = folium.Marker(location=[record["Lat"], record["Long"]],
                            icon=folium.Icon(icon="star",color='white', icon_color=record['marker_color']),
                            popup=f"{record['Launch Site']} - {pop_up(record['class'])}")
        marker_cluster.add_child(marker)
    formatter = "function(num) {return L.Util.formatNum(num, 5);};"
    mouse_position = MousePosition(
        position='topright',
        separator=' Long: ',
        empty_string='NaN',
        lng_first=False,
        num_digits=20,
        prefix='Lat:',
        lat_formatter=formatter,
        lng_formatter=formatter,
    )

    site_map.add_child(mouse_position)
    sites = spacex_df.groupby("Launch Site",as_index=False)[['Lat','Long']].mean()
    for index,record in sites.iterrows():
        coastline_coords = [28.56315,-80.56797]
        if record['Launch Site'] == "VAFB SLC-4E":
            coastline_coords = [34.63243,-120.62659]
        distance_coastline = calculate_distance(record['Lat'], record['Long'], coastline_coords[0], coastline_coords[1])
        distance_marker = folium.Marker(
        coastline_coords,
        icon=DivIcon(
            icon_size=(20,20),
            icon_anchor=(0,0),
            html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % "{:10.2f} KM".format(distance_coastline),
            )
        )
        line=folium.PolyLine(locations=[(record['Lat'], record['Long']),tuple(coastline_coords)], weight=1)
        marker_cluster.add_child(distance_marker)
        site_map.add_child(line)
    
    return site_map

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))
def get_pie_chart(entered_site):
    filtered_df = spacex_df_plot
    if entered_site == 'ALL':
        fig = px.pie(data_frame=filtered_df, values='class', 
                     names='Launch Site', 
                     title='Total Success Launches by Site')
    else:
        filtered_df = spacex_df_plot[spacex_df_plot['Launch Site'] == entered_site]
        fig = px.pie(data_frame=filtered_df, 
                     names='class', 
                     title=f'Success and Fail Count for site {entered_site}')
    return fig

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id='success-payload-scatter-chart', component_property='figure'),
              [Input(component_id='site-dropdown', component_property='value'),
               Input(component_id='payload-slider', component_property='value')])
def get_scatter_chart(entered_site, payload_range):
    filtered_df = spacex_df_plot[(spacex_df_plot["Payload Mass (kg)"] >= payload_range[0]) &
                            (spacex_df_plot["Payload Mass (kg)"] <= payload_range[1])]
    if entered_site == 'ALL':
        fig = px.scatter(data_frame=filtered_df, x="Payload Mass (kg)", y="class", color="Booster Version",
                         title='Correlation between Payload and Success for All Sites')
    else:
        filtered_df = filtered_df[filtered_df['Launch Site'] == entered_site]
        fig = px.scatter(data_frame=filtered_df, x="Payload Mass (kg)", y="class", color="Booster Version",
                         title=f'Correlation Between Payload and Success for site {entered_site}')
    return fig

@app.callback(
    Output('folium-map', 'srcDoc'),
    [Input('site-dropdown', 'value')]
)
def update_map(entered_site):
    folium_map = create_folium_map()
    return folium_map._repr_html_()


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
