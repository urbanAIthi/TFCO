import plotly as py
import plotly.graph_objs as go
import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from create_fco_target import create_fco_target
import plotly.express as px
import plotly.figure_factory as ff
import plotly.subplots as sp
import pandas as pd

# get the config
from configs.config_fco_analysis import config as an_cfg

def run_fco_analysis(cfg):
    # create a new dir to save the analysis
    #os.mkdir(os.path.join('fco_analyzation', cfg['dataset_name']))
    # copy the current config to the analysis dir
    #with open(os.path.join('fco_analyzation', cfg['dataset_name'], 'config.py'), 'w') as f:
    #    f.write(f'config = {cfg}')
    
    # get the base dataset
    base_dataset = cfg['base_dataset']

    # get the sequence length that should be checked
    sequence_len = cfg['sequence_len'] # list

    # get the pen rates that should be checked
    pen_rates = cfg['pen_rates'] # list

    # get path to all the datasets given the pen_rates and base_dataset
    dataset_paths = [base_dataset.replace('xx', str(pen_rate)) for pen_rate in pen_rates]

    sequence_path = [f'detected_vehicles_time_{seq_len}' for seq_len in sequence_len]

    full_path_grid = [[os.path.join(dataset_path, seq_path) for seq_path in sequence_path] for dataset_path in dataset_paths]

    # iterate through the grid and create the fco target if it does not exist
    for path_grid in full_path_grid:
        for path in path_grid:
            path = os.path.join('data', path)
            if not os.path.exists(path):
                # current sequence length
                seq_len = int(path.split('_')[-1])
                create_fco_target(path[0:path.rfind('/')], seq_len)
            else:
                print(f'fco target already exists for {path}')


    
    # create data dict with all possible combinations in the grid
    data_dict = dict()
    for i, path_grid in enumerate(full_path_grid):
        for j, path in enumerate(path_grid):
            # load the data
            fco_time_diff=np.load(os.path.join('data',path, 'Analysis', 'fco_time_diff.npy'))
            fco_complete_diff=np.load(os.path.join('data',path, 'Analysis', 'fco_complete_diff.npy'))
            complete=np.load(os.path.join('data',path, 'Analysis', 'complete.npy'))
            current = np.load(os.path.join('data',path, 'Analysis', 'fco_current.npy'))
            fco_time = np.load(os.path.join('data',path, 'Analysis', 'fco_time.npy'))
            

            fco_diff =current/complete# calculate lost
            fco_diff = fco_diff[~np.isnan(fco_diff)]
            fco_time_diff = fco_time/complete # calculate recovered
            fco_time_diff = fco_time_diff[~np.isnan(fco_time_diff)]
            complete = complete[complete != 0]
            data_dict[(pen_rates[i], sequence_len[j])] = {'sequence_len': sequence_len[j],
                                                           'penetration_rate': pen_rates[i], 
                                                           'fco': fco_diff,
                                                            'fco_time': fco_time_diff,
                                                            'complete': complete}

    
    # create dist grid plot for the fco_diff
    create_grid_plot(data_dict, 'fco', 'dist', None)
    create_grid_plot(data_dict, 'fco_time', 'dist', None)
    create_grid_plot(data_dict, 'fco', 'scatter', None)
    create_grid_plot(data_dict, 'fco_time', 'scatter', None)
    create_grid_plot(data_dict, 'fco', 'line', None)
    create_grid_plot(data_dict, 'fco_time', 'line', None)

    # iterate through the grid and get the Analysis data
    fco_time_diff_grid = [] # diff from fco last sequences to all current frame
    fco_complete_diff_grid = [] # diff from fco current frame to all last sequences
    complete_grid = [] # all current frame
    fco_current_diff_grid = [] # diff from fco current frame to all current frame

    mean_fco_time_diff_grid = []
    mean_fco_complete_diff_grid = []
    mean_complete_grid = []
    mean_fco_current_diff_grid = []

    rel_mean_fco_time_diff_grid = []
    rel_mean_fco_complete_diff_grid = []
    rel_mean_fco_current_diff_grid = []


    for path_grid in full_path_grid:
        fco_time_diff_grid.append([])
        fco_complete_diff_grid.append([])
        complete_grid.append([])

        mean_fco_time_diff_grid.append([])
        mean_fco_complete_diff_grid.append([])
        mean_complete_grid.append([])

        rel_mean_fco_time_diff_grid.append([])
        rel_mean_fco_complete_diff_grid.append([])
        rel_mean_fco_current_diff_grid.append([])

        for path in path_grid:
            path = os.path.join('data', path)
            fco_time_diff=np.load(os.path.join(path, 'Analysis', 'fco_time_diff.npy'))
            fco_complete_diff=np.load(os.path.join(path, 'Analysis', 'fco_complete_diff.npy'))
            complete=np.load(os.path.join(path, 'Analysis', 'complete.npy'))
            current = np.load(os.path.join(path, 'Analysis', 'fco_current.npy'))

            fco_time_diff_grid[-1].append(fco_time_diff)
            fco_complete_diff_grid[-1].append(fco_complete_diff)
            complete_grid[-1].append(complete)

            mean_fco_time_diff_grid[-1].append(np.mean(fco_time_diff))
            mean_fco_complete_diff_grid[-1].append(np.mean(fco_complete_diff))
            mean_complete_grid[-1].append(np.mean(complete))

            rel_mean_fco_time_diff_grid[-1].append(np.mean(fco_time_diff)/np.mean(complete))
            rel_mean_fco_complete_diff_grid[-1].append(np.mean(fco_complete_diff)/np.mean(complete))
            rel_mean_fco_current_diff_grid[-1].append((np.mean(complete)-np.mean(fco_complete_diff))/np.mean(complete))
    
    # reduce precision rel_mean_fco_time_diff_grid to 2 digits after comma
    rel_mean_fco_time_diff_grid = np.round(rel_mean_fco_time_diff_grid, 2)


    fig = px.imshow(
        rel_mean_fco_time_diff_grid,
        x=sequence_len,
        y=pen_rates,  # Use the reversed pen_rates
        text_auto=True,  # Disable the color scale
    )

    fig.update_xaxes(title_text='Sequence Length', tickvals=[5,10,15,20])  # Replace 'X Label' with your desired x-axis label
    fig.update_yaxes(title_text='FCO Penetration Rate', tickvals=[5,10,20,30])  # Replace 'Y Label' with your desired y-axis label
    fig.update_layout(
        autosize=False,
        width=750,  # Adjust the width as needed
        height=500,  # Adjust the height as needed
    )
    fig.update_coloraxes(showscale=False)
    fig.write_image('test.png', scale=5)

    rel_mean_fcu_current_diff_grid = np.round(rel_mean_fco_current_diff_grid, 2)
    fig = px.imshow(
        rel_mean_fcu_current_diff_grid,
        x=sequence_len,
        y=pen_rates, 
        text_auto=True, 
    )
    fig.update_xaxes(title_text='', tickvals=[])  # Replace 'X Label' with your desired x-axis label
    fig.update_yaxes(title_text='FCO Penetration Rate', tickvals=[5,10,20,30])  # Replace 'Y Label' with your desired y-axis label
    fig.update_layout(
        autosize=False,
        width=750,  # Adjust the width as needed
        height=500,  # Adjust the height as needed
    )
    fig.update_coloraxes(showscale=False)
    fig.write_image('test_current.png', scale=5)


def create_grid_plot(items: dict, data_type: str, plot_type: str, path):
    # Extract unique sequence_len and penetration_rate
    unique_sequence_lens = sorted(set(item['sequence_len'] for item in items.values()))
    unique_pen_rates = sorted(set(item['penetration_rate'] for item in items.values()))

    # Function to add distplot and customize x-axis to show only the mean
    def add_distplot_with_custom_xaxis(fig, data, row, col):
        mean = np.mean(data)
        distplot = ff.create_distplot([data], [''], show_rug=False, show_hist=False)
        for trace in distplot['data']:
            fig.add_trace(trace, row=row, col=col)
        # Customize x-axis to show only the mean value
        fig.update_xaxes(showgrid=False, tickmode='array', tickvals=[mean], ticktext=[f'Mean: {mean:.2f}'], row=row, col=col)
        fig.update_yaxes(showgrid=False, tickvals=[])

    # Function to add scatter plot
    def add_scatter_plot(fig, data, complete, row, col):
        fig.add_trace(go.Scatter(x=complete, y=data, mode='markers'), row=row, col=col)
        fig.update_xaxes(title_text="Complete", row=row, col=col)
        fig.update_yaxes(title_text=f"{data_type}/Complete", row=row, col=col)

    # Function to add line plot
    def add_line_plot(fig, current_complete, complete, row, col):
        # Group by 'complete' and calculate mean
        df = pd.DataFrame({'complete': complete, 'current_complete': current_complete})
        mean_values = df.groupby('complete').mean().reset_index()

        # Add line plot
        fig.add_trace(go.Scatter(x=mean_values['complete'], y=mean_values['current_complete'], mode='lines'), row=row, col=col)
        fig.update_xaxes(title_text="Complete", row=row, col=col)
        fig.update_yaxes(title_text=f"Mean of {data_type}/Complete", row=row, col=col)

    # Create subplot grid
    fig = sp.make_subplots(rows=len(unique_sequence_lens), cols=len(unique_pen_rates), 
                        subplot_titles=[f"sequence_len: {seq_len} penetration_rate: {pen_rate}" 
                                        for seq_len in unique_sequence_lens 
                                        for pen_rate in unique_pen_rates])

    # Create a mapping from (sequence_len, penetration_rate) to (row, col)
    row_col_mapping = {(seq_len, pen_rate): (unique_sequence_lens.index(seq_len) + 1, unique_pen_rates.index(pen_rate) + 1) 
                    for seq_len in unique_sequence_lens 
                    for pen_rate in unique_pen_rates}

    if plot_type == 'dist':
        # Add each distplot to the subplot
        for key, item in items.items():
            row, col = row_col_mapping[(item['sequence_len'], item['penetration_rate'])]
            add_distplot_with_custom_xaxis(fig, item[data_type], row, col)
    
    elif plot_type == 'scatter':
        # Add each scatter plot to the subplot
        for key, item in items.items():
            row, col = row_col_mapping[(item['sequence_len'], item['penetration_rate'])]
            add_scatter_plot(fig, item[data_type], item['complete'], row, col)
    
    elif plot_type == 'line':
        # Add each line plot to the subplot
        for key, item in items.items():
            row, col = row_col_mapping[(item['sequence_len'], item['penetration_rate'])]
            add_line_plot(fig, item[data_type], item['complete'], row, col)
    else:
        raise ValueError(f'plot_type {plot_type} is not supported')

    # Update layout if needed
    fig.update_layout(height=600, width=800)
    # save the plot as svg
    fig.write_image(f'{data_type}_{plot_type}.svg', scale=5)







if __name__ == '__main__':
    run_fco_analysis(an_cfg)