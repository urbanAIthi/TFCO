from typing import List
import os
import ast
import pandas as pd
import plotly.graph_objects as go
color_scale = [[0.25*i, color] for i, color in enumerate(['#D3E5FF', '#A1C9FD', '#5A8FDB', '#2E5A88'])]

def create_grid_plot(gid_mean_values: List[List]):
    fig = go.Figure(data=go.Heatmap(
                   z=gid_mean_values,
                   x=['s=5s', 's=10s', 's=15s', 's=20s'],
                   y=['p=5%', 'p=10%', 'p=15%', 'p=20%'],
                   text=[[str(round(value, 2)) for value in row] for row in gid_mean_values],
                   texttemplate="%{text}",
                   showscale=False,
                   colorscale='Blues'
    ))

    fig.update_layout(
        xaxis_nticks=36,
        yaxis_nticks=36,
        xaxis=dict(
            tickfont=dict(size=16)  # Increase x-axis font size
        ),
        yaxis=dict(
            tickfont=dict(size=16)  # Increase y-axis font size
        )
    )
    
    fig.update_traces(textfont=dict(size=16))
    
    fig.write_image("time_potential_grid.png")

if __name__ == "__main__":
    analyzation_path = ['i3040_newdetector_50p', 'i3040_newdetector_100p', 'i3040_newdetector_150p' ,'i3040_newdetector_200p']
    pen_rates = [5, 10, 15, 20]
    sequence_len = [5, 10, 15, 20]

    gid_mean_values = []#[[0.09101509596621644, 0.13948658152511828, 0.17175013897774327, 0.19609691186580155], [0.10269125717018136, 0.15019584564963406, 0.1804509320108257, 0.20192423707014462], [0.09847280942297955, 0.13942155826459268, 0.1634228117001455, 0.1791919060290385], [0.08268338944039064, 0.11093966492826213, 0.12680141678447845, 0.1366167470078681]]
    if len(gid_mean_values) == 0:
        for path in analyzation_path:
            current_grid = []
            for seq_len in sequence_len:
                # load the dataset
                dataset = pd.read_csv(os.path.join('data', path, 'dataset.csv'))
                dataset['complete_vehicle_infos'] = dataset['complete_vehicle_infos'].apply(ast.literal_eval)
                dataset['detected_vehicle_infos'] = dataset['detected_vehicle_infos'].apply(ast.literal_eval)
                dataset['fco_vehicle_infos'] = dataset['fco_vehicle_infos'].apply(ast.literal_eval)
                dataset[f'detected_time_{seq_len}_vehicle_infos'] = dataset[f'detected_time_{seq_len}_vehicle_infos'].apply(ast.literal_eval)

                # create new column with all vehicles that are in the fco or detected without duplicates
                dataset['fco_or_detected'] = dataset.apply(lambda x: list(set(list(x['fco_vehicle_infos'].keys()) + list(x['detected_vehicle_infos'].keys()))), axis=1)
                dataset['len_fco_or_detected'] = dataset['fco_or_detected'].apply(lambda x: len(x))

                dataset['len_complete'] = dataset['complete_vehicle_infos'].apply(lambda x: len(x))
                dataset['len_time'] = dataset[f'detected_time_{seq_len}_vehicle_infos'].apply(lambda x: len(x))

                dataset['time_potential'] = (dataset['len_time'] - dataset['len_fco_or_detected'] )/ dataset['len_complete']

                current_grid.append(dataset['time_potential'].mean())
            gid_mean_values.append(current_grid)
        print(gid_mean_values)
    create_grid_plot(gid_mean_values)